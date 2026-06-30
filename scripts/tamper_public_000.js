const fs = require("fs");

const publicPath = "build/batch/public_000.json";
const tamperedPath = "build/batch/public_000_tampered.json";

const pub = JSON.parse(fs.readFileSync(publicPath, "utf8"));

// public input order:
// [nonce, modelHash, inputHash, clientID, requestHash, normalScore, attackScore]

// Tamper attackScore
pub[6] = (BigInt(pub[6]) + 1n).toString();

fs.writeFileSync(tamperedPath, JSON.stringify(pub, null, 2));

console.log("Tampered public file created:");
console.log(tamperedPath);
console.log("Changed attackScore public value.");