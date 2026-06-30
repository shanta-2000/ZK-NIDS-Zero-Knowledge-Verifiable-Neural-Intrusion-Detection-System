const fs = require("fs");
const circomlibjs = require("circomlibjs");

async function main() {
    const poseidon = await circomlibjs.buildPoseidon();
    const F = poseidon.F;

    function poseidonHash(arr) {
        return F.toString(poseidon(arr.map((x) => BigInt(x))));
    }

    // -------------------------
    // Private traffic features
    // Example network traffic sample
    // -------------------------
    const x = [20, 80, 65, 90, 40, 30];

    // -------------------------
    // Private model weights
    // Small neural network:
    // 6 input → 4 hidden → 2 output
    // -------------------------
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

    // -------------------------
    // Private client credential data
    // -------------------------
    const clientCertificate = 123456;
    const salt = 78910;

    // Public nonce
    const nonce = 1;

    // -------------------------
    // Flatten model values in the SAME order as circuit
    // W1 → b1 → W2 → b2
    // Total = 38 values
    // -------------------------
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

    // -------------------------
    // Chunked model hash
    // Must match circuit:
    // chunk1 = first 16
    // chunk2 = next 16
    // chunk3 = last 6
    // modelHash = Poseidon(hash1, hash2, hash3)
    // -------------------------
    const chunk1 = modelValues.slice(0, 16);
    const chunk2 = modelValues.slice(16, 32);
    const chunk3 = modelValues.slice(32, 38);

    const hash1 = poseidonHash(chunk1);
    const hash2 = poseidonHash(chunk2);
    const hash3 = poseidonHash(chunk3);

    const modelHash = poseidonHash([hash1, hash2, hash3]);

    // -------------------------
    // Other public hashes
    // -------------------------
    const inputHash = poseidonHash(x);
    const clientID = poseidonHash([clientCertificate, salt]);
    const requestHash = poseidonHash([clientID, inputHash, nonce]);

    // -------------------------
    // Same neural network calculation as circuit
    // hidden[j] = raw[j]^2
    // -------------------------
    const hidden = [];

    for (let j = 0; j < 4; j++) {
        let raw = BigInt(b1[j]);

        for (let i = 0; i < 6; i++) {
            raw += BigInt(W1[j][i]) * BigInt(x[i]);
        }

        hidden[j] = raw * raw;
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

    console.log("input/input.json created successfully.");
    console.log("normalScore =", normalScore);
    console.log("attackScore =", attackScore);

    if (BigInt(attackScore) > BigInt(normalScore)) {
        console.log("Prediction: Attack");
    } else {
        console.log("Prediction: Normal");
    }

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