pragma circom 2.1.6;

include "../node_modules/circomlib/circuits/poseidon.circom";

template ZKNIDS() {
    // Private traffic features
    signal input x[6];

    // Private neural network weights
    signal input W1[4][6];
    signal input b1[4];
    signal input W2[2][4];
    signal input b2[2];

    // Private client credential data
    signal input clientCertificate;
    signal input salt;

    // Public values
    signal input nonce;
    signal input modelHash;
    signal input inputHash;
    signal input clientID;
    signal input requestHash;
    signal input normalScore;
    signal input attackScore;

    // =========================================================
    // 1. Model hash check using chunked Poseidon
    //
    // Total model values:
    // W1 = 4*6 = 24
    // b1 = 4
    // W2 = 2*4 = 8
    // b2 = 2
    // Total = 38
    //
    // Poseidon(38) gives error, so we split:
    // chunk1 = 16 values
    // chunk2 = 16 values
    // chunk3 = 6 values
    // final modelHash = Poseidon(chunkHash1, chunkHash2, chunkHash3)
    // =========================================================

    component mh1 = Poseidon(16);
    component mh2 = Poseidon(16);
    component mh3 = Poseidon(6);
    component mhFinal = Poseidon(3);

    // Flattened model order:
    // W1 values first = 24 values
    // b1 next = 4 values
    // W2 next = 8 values
    // b2 last = 2 values

    // chunk 1: first 16 values from W1
    mh1.inputs[0] <== W1[0][0];
    mh1.inputs[1] <== W1[0][1];
    mh1.inputs[2] <== W1[0][2];
    mh1.inputs[3] <== W1[0][3];
    mh1.inputs[4] <== W1[0][4];
    mh1.inputs[5] <== W1[0][5];

    mh1.inputs[6] <== W1[1][0];
    mh1.inputs[7] <== W1[1][1];
    mh1.inputs[8] <== W1[1][2];
    mh1.inputs[9] <== W1[1][3];
    mh1.inputs[10] <== W1[1][4];
    mh1.inputs[11] <== W1[1][5];

    mh1.inputs[12] <== W1[2][0];
    mh1.inputs[13] <== W1[2][1];
    mh1.inputs[14] <== W1[2][2];
    mh1.inputs[15] <== W1[2][3];

    // chunk 2: remaining W1 + b1 + part of W2
    mh2.inputs[0] <== W1[2][4];
    mh2.inputs[1] <== W1[2][5];

    mh2.inputs[2] <== W1[3][0];
    mh2.inputs[3] <== W1[3][1];
    mh2.inputs[4] <== W1[3][2];
    mh2.inputs[5] <== W1[3][3];
    mh2.inputs[6] <== W1[3][4];
    mh2.inputs[7] <== W1[3][5];

    mh2.inputs[8] <== b1[0];
    mh2.inputs[9] <== b1[1];
    mh2.inputs[10] <== b1[2];
    mh2.inputs[11] <== b1[3];

    mh2.inputs[12] <== W2[0][0];
    mh2.inputs[13] <== W2[0][1];
    mh2.inputs[14] <== W2[0][2];
    mh2.inputs[15] <== W2[0][3];

    // chunk 3: remaining W2 + b2
    mh3.inputs[0] <== W2[1][0];
    mh3.inputs[1] <== W2[1][1];
    mh3.inputs[2] <== W2[1][2];
    mh3.inputs[3] <== W2[1][3];
    mh3.inputs[4] <== b2[0];
    mh3.inputs[5] <== b2[1];

    mhFinal.inputs[0] <== mh1.out;
    mhFinal.inputs[1] <== mh2.out;
    mhFinal.inputs[2] <== mh3.out;

    modelHash === mhFinal.out;

    // =========================================================
    // 2. Input hash check
    // inputHash = Poseidon(x[0],...,x[5])
    // =========================================================

    component ih = Poseidon(6);

    for (var i = 0; i < 6; i++) {
        ih.inputs[i] <== x[i];
    }

    inputHash === ih.out;

    // =========================================================
    // 3. Client ID check
    // clientID = Poseidon(clientCertificate, salt)
    // =========================================================

    component cid = Poseidon(2);
    cid.inputs[0] <== clientCertificate;
    cid.inputs[1] <== salt;

    clientID === cid.out;

    // =========================================================
    // 4. Request hash check
    // requestHash = Poseidon(clientID, inputHash, nonce)
    // =========================================================

    component rh = Poseidon(3);
    rh.inputs[0] <== clientID;
    rh.inputs[1] <== inputHash;
    rh.inputs[2] <== nonce;

    requestHash === rh.out;

    // =========================================================
    // 5. Neural Network Inference
    //
    // 6 input features → 4 hidden neurons → 2 output scores
    // hidden[j] = raw[j]^2
    // =========================================================

    // =========================================================
    // 5. Neural Network Inference
    //
    // Circom does not allow many multiplications inside one line.
    // So each multiplication is calculated separately first.
    // =========================================================

    signal p00;
    signal p01;
    signal p02;
    signal p03;
    signal p04;
    signal p05;

    signal p10;
    signal p11;
    signal p12;
    signal p13;
    signal p14;
    signal p15;

    signal p20;
    signal p21;
    signal p22;
    signal p23;
    signal p24;
    signal p25;

    signal p30;
    signal p31;
    signal p32;
    signal p33;
    signal p34;
    signal p35;

    p00 <== W1[0][0] * x[0];
    p01 <== W1[0][1] * x[1];
    p02 <== W1[0][2] * x[2];
    p03 <== W1[0][3] * x[3];
    p04 <== W1[0][4] * x[4];
    p05 <== W1[0][5] * x[5];

    p10 <== W1[1][0] * x[0];
    p11 <== W1[1][1] * x[1];
    p12 <== W1[1][2] * x[2];
    p13 <== W1[1][3] * x[3];
    p14 <== W1[1][4] * x[4];
    p15 <== W1[1][5] * x[5];

    p20 <== W1[2][0] * x[0];
    p21 <== W1[2][1] * x[1];
    p22 <== W1[2][2] * x[2];
    p23 <== W1[2][3] * x[3];
    p24 <== W1[2][4] * x[4];
    p25 <== W1[2][5] * x[5];

    p30 <== W1[3][0] * x[0];
    p31 <== W1[3][1] * x[1];
    p32 <== W1[3][2] * x[2];
    p33 <== W1[3][3] * x[3];
    p34 <== W1[3][4] * x[4];
    p35 <== W1[3][5] * x[5];

    signal raw0;
    signal raw1;
    signal raw2;
    signal raw3;

    raw0 <== p00 + p01 + p02 + p03 + p04 + p05 + b1[0];
    raw1 <== p10 + p11 + p12 + p13 + p14 + p15 + b1[1];
    raw2 <== p20 + p21 + p22 + p23 + p24 + p25 + b1[2];
    raw3 <== p30 + p31 + p32 + p33 + p34 + p35 + b1[3];

    signal h0;
    signal h1;
    signal h2;
    signal h3;

    h0 <== raw0 * raw0;
    h1 <== raw1 * raw1;
    h2 <== raw2 * raw2;
    h3 <== raw3 * raw3;

    signal q00;
    signal q01;
    signal q02;
    signal q03;

    signal q10;
    signal q11;
    signal q12;
    signal q13;

    q00 <== W2[0][0] * h0;
    q01 <== W2[0][1] * h1;
    q02 <== W2[0][2] * h2;
    q03 <== W2[0][3] * h3;

    q10 <== W2[1][0] * h0;
    q11 <== W2[1][1] * h1;
    q12 <== W2[1][2] * h2;
    q13 <== W2[1][3] * h3;

    signal computedNormalScore;
    signal computedAttackScore;

    computedNormalScore <== q00 + q01 + q02 + q03 + b2[0];
    computedAttackScore <== q10 + q11 + q12 + q13 + b2[1];

    normalScore === computedNormalScore;
    attackScore === computedAttackScore;
}

component main { public [
    nonce,
    modelHash,
    inputHash,
    clientID,
    requestHash,
    normalScore,
    attackScore
] } = ZKNIDS();