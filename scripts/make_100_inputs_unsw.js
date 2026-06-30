const fs = require("fs");
const path = require("path");
const circomlibjs = require("circomlibjs");

async function main() {
    const poseidon = await circomlibjs.buildPoseidon();
    const F = poseidon.F;

    function poseidonHash(arr) {
        const inputs = arr.map((x) => F.e(BigInt(x)));
        return F.toString(poseidon(inputs));
    }

    function computeScores(x, W1, b1, W2, b2) {
        const hidden = [];

        for (let j = 0; j < 4; j++) {
            let raw = BigInt(b1[j]);

            for (let i = 0; i < 6; i++) {
                raw += BigInt(W1[j][i]) * BigInt(x[i]);
            }

            hidden[j] = raw * raw;
        }

        const scores = [];

        for (let k = 0; k < 2; k++) {
            let score = BigInt(b2[k]);

            for (let j = 0; j < 4; j++) {
                score += BigInt(W2[k][j]) * hidden[j];
            }

            scores[k] = score;
        }

        return {
            normalScore: scores[0].toString(),
            attackScore: scores[1].toString(),
        };
    }

    const model = JSON.parse(fs.readFileSync("model/trained_model.json", "utf8"));
    const rows = JSON.parse(fs.readFileSync("data/selected_100_rows.json", "utf8"));

    const W1 = model.W1;
    const b1 = model.b1;
    const W2 = model.W2;
    const b2 = model.b2;

    fs.mkdirSync("input/batch", { recursive: true });
    fs.mkdirSync("results", { recursive: true });

    // Model hash: Hash(W1, b1, W2, b2)
    // Same order as the Circom circuit
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

    const clientCertificate = 123456;
    const salt = 78910;
    const clientID = poseidonHash([clientCertificate, salt]);

    const summaryRows = [];

    for (const row of rows) {
        const sampleId = row.sample_id;
        const x = row.x;
        const nonce = sampleId + 1;

        const inputHash = poseidonHash(x);
        const requestHash = poseidonHash([clientID, inputHash, nonce]);

        const { normalScore, attackScore } = computeScores(x, W1, b1, W2, b2);

        const prediction = BigInt(attackScore) > BigInt(normalScore)
            ? "Attack"
            : "Normal";

        const input = {
            x: x.map(String),
            W1: W1.map((r) => r.map(String)),
            b1: b1.map(String),
            W2: W2.map((r) => r.map(String)),
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

        const id = String(sampleId).padStart(3, "0");
        fs.writeFileSync(
            path.join("input/batch", `input_${id}.json`),
            JSON.stringify(input, null, 2)
        );

        summaryRows.push({
            sample_id: sampleId,
            dataset_label: row.dataset_label,
            attack_cat: row.attack_cat,
            prediction,
            correct: prediction === row.dataset_label,
            normalScore,
            attackScore,
            inputHash,
            requestHash,
        });
    }

    const csvHeader = [
        "sample_id",
        "dataset_label",
        "attack_cat",
        "prediction",
        "correct",
        "normalScore",
        "attackScore",
        "inputHash",
        "requestHash",
    ];

    const csvLines = [csvHeader.join(",")];

    for (const r of summaryRows) {
        csvLines.push(csvHeader.map((h) => r[h]).join(","));
    }

    fs.writeFileSync("results/generated_100_inputs_summary.csv", csvLines.join("\n"));

    const correctCount = summaryRows.filter((r) => r.correct).length;

    console.log("100 dataset-based input JSON files generated.");
    console.log("Location: input/batch/input_000.json ... input_099.json");
    console.log("Model hash:", modelHash);
    console.log("ML correct predictions:", `${correctCount}/100`);
    console.log("Summary saved: results/generated_100_inputs_summary.csv");
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});