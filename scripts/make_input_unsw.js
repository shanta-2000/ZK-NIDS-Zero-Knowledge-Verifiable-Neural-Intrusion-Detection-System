const fs = require("fs");
const circomlibjs = require("circomlibjs");

async function main() {
    const poseidon = await circomlibjs.buildPoseidon();
    const F = poseidon.F;

    function poseidonHash(arr) {
        return F.toString(poseidon(arr.map((x) => BigInt(x))));
    }

    function scaleTo100(value, cap) {
        let v = Number(value);
        if (!Number.isFinite(v) || v < 0) v = 0;
        if (v > cap) v = cap;
        return Math.round((v / cap) * 100);
    }

    // =====================================================
    // Dataset selection
    // Use training set first
    // rowIndex = 47911 selects an Attack sample in training set
    // You can later change it to test other rows
    // =====================================================

    const datasetPath = "data/UNSW_NB15_training-set.csv";
    const rowIndex = 47911;

    const csv = fs.readFileSync(datasetPath, "utf8").trim();
    const lines = csv.split(/\r?\n/);

    const header = lines[0].split(",");
    const row = lines[rowIndex + 1].split(",");

    function getValue(columnName) {
        const idx = header.indexOf(columnName);
        if (idx === -1) {
            throw new Error(`Column not found: ${columnName}`);
        }
        return row[idx];
    }

    const rawDur = getValue("dur");
    const rawSpkts = getValue("spkts");
    const rawDpkts = getValue("dpkts");
    const rawSbytes = getValue("sbytes");
    const rawDbytes = getValue("dbytes");
    const rawRate = getValue("rate");

    const attackCat = getValue("attack_cat");
    const label = getValue("label"); // 0 = Normal, 1 = Attack

    // =====================================================
    // Normalize selected UNSW-NB15 features to 0–100 integers
    // These are private traffic features for the circuit
    // =====================================================

    const x = [
        scaleTo100(rawDur, 10),       // duration capped at 10 sec
        scaleTo100(rawSpkts, 100),    // source packets capped at 100
        scaleTo100(rawDpkts, 100),    // destination packets capped at 100
        scaleTo100(rawSbytes, 10000), // source bytes capped at 10000
        scaleTo100(rawDbytes, 10000), // destination bytes capped at 10000
        scaleTo100(rawRate, 200000),  // rate capped at 200000
    ];

    // =====================================================
    // Demo neural network weights
    // Same architecture as our circuit:
    // 6 input → 4 hidden → 2 output
    // =====================================================

    const W1 = [
        [1, 2, 1, 1, 0, 1],
        [2, 1, 0, 1, 1, 0],
        [0, 1, 2, 1, 1, 1],
        [1, 0, 1, 2, 1, 1],
    ];

    const b1 = [3, 2, 1, 4];

    const W2 = [
        [1, 1, 0, 1], // normal score weights
        [2, 1, 3, 2], // attack score weights
    ];

    const b2 = [5, 7];

    // =====================================================
    // Private client credential data
    // =====================================================

    const clientCertificate = 123456;
    const salt = 78910;

    // Use dataset row as nonce so every row has a unique request
    const nonce = rowIndex + 1;

    // =====================================================
    // Model hash: Hash(W1, b1, W2, b2)
    // Must match chunked hash structure in our circuit
    // =====================================================

    const modelValues = [];

    for (let j = 0; j < 4; j++) {
        for (let i = 0; i < 6; i++) {
            modelValues.push(W1[j][i]);
        }
    }

    for (let j = 0; j < 4; j++) {
        modelValues.push(b1[j]);
    }

    for (let k = 0; k < 2; k++) {
        for (let j = 0; j < 4; j++) {
            modelValues.push(W2[k][j]);
        }
    }

    for (let k = 0; k < 2; k++) {
        modelValues.push(b2[k]);
    }

    const chunk1 = modelValues.slice(0, 16);
    const chunk2 = modelValues.slice(16, 32);
    const chunk3 = modelValues.slice(32, 38);

    const hash1 = poseidonHash(chunk1);
    const hash2 = poseidonHash(chunk2);
    const hash3 = poseidonHash(chunk3);

    const modelHash = poseidonHash([hash1, hash2, hash3]);

    // =====================================================
    // Input/client/request hashes
    // =====================================================

    const inputHash = poseidonHash(x);
    const clientID = poseidonHash([clientCertificate, salt]);
    const requestHash = poseidonHash([clientID, inputHash, nonce]);

    // =====================================================
    // Neural network inference outside circuit
    // These values must match circuit computation
    // =====================================================

    const hidden = [];

    for (let j = 0; j < 4; j++) {
        let raw = BigInt(b1[j]);

        for (let i = 0; i < 6; i++) {
            raw += BigInt(W1[j][i]) * BigInt(x[i]);
        }

        hidden[j] = raw * raw; // square activation
    }

    const out = [];

    for (let k = 0; k < 2; k++) {
        let score = BigInt(b2[k]);

        for (let j = 0; j < 4; j++) {
            score += BigInt(W2[k][j]) * hidden[j];
        }

        out[k] = score;
    }

    const normalScore = out[0].toString();
    const attackScore = out[1].toString();

    const prediction = BigInt(attackScore) > BigInt(normalScore)
        ? "Attack"
        : "Normal";

    const datasetLabel = label === "1" ? "Attack" : "Normal";

    // =====================================================
    // Final input.json for Circom witness generation
    // =====================================================

    const input = {
        x: x.map(String),
        W1: W1.map((row) => row.map(String)),
        b1: b1.map(String),
        W2: W2.map((row) => row.map(String)),
        b2: b2.map(String),

        clientCertificate: clientCertificate.toString(),
        salt: salt.toString(),

        nonce: nonce.toString(),
        modelHash,
        inputHash,
        clientID,
        requestHash,
        normalScore,
        attackScore,
    };

    fs.mkdirSync("input", { recursive: true });
    fs.writeFileSync("input/input.json", JSON.stringify(input, null, 2));

    console.log("UNSW-NB15 dataset input generated successfully.");
    console.log("Dataset file:", datasetPath);
    console.log("Selected row index:", rowIndex);
    console.log("Attack category:", attackCat);
    console.log("Dataset label:", datasetLabel);
    console.log("Raw selected features:");
    console.log({
        dur: rawDur,
        spkts: rawSpkts,
        dpkts: rawDpkts,
        sbytes: rawSbytes,
        dbytes: rawDbytes,
        rate: rawRate,
    });
    console.log("Normalized circuit input x:", x);
    console.log("normalScore =", normalScore);
    console.log("attackScore =", attackScore);
    console.log("Model prediction:", prediction);
    console.log("Prediction matches dataset label?:", prediction === datasetLabel ? "Yes" : "No");

    console.log("\nPublic values:");
    console.log("modelHash   =", modelHash);
    console.log("inputHash   =", inputHash);
    console.log("clientID    =", clientID);
    console.log("requestHash =", requestHash);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});