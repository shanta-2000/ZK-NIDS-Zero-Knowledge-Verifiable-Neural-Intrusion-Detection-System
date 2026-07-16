# ZK-NIDS: Zero-Knowledge Verifiable Neural Intrusion Detection System.
The main idea of this project is to combine AI-based network intrusion detection with zero-knowledge proof, so that a prediction can be verified without revealing sensitive traffic data or private model information.
## Background

A Network Intrusion Detection System, or NIDS, monitors network traffic to detect suspicious or malicious activities.
Traditional NIDS usually depends on fixed rules or known attack signatures.
However, modern cyberattacks are more dynamic, so AI and machine learning models are now widely used for intrusion detection.
In an AI-based NIDS, network traffic features are given to an AI model, and the model predicts whether the traffic is normal or an attack.

## Proposed system is called ZK-NIDS.
It combines AI-based intrusion detection with zero-knowledge proof.
The system takes hidden traffic input and hidden neural model weights, computes normal and attack scores, and then generates a zkSNARK proof.
The verifier can check the proof and confirm that the prediction was honestly computed, without knowing the private traffic features, private model weights, or client certificate.

## Dataset

For our experiment, we used the public UNSW-NB15 intrusion detection dataset.
We used the training CSV file with 175,341 records for model training and preparation.
We used the testing CSV file with 82,332 records for testing and ZK proof generation.
For the final experiment, we selected 100 balanced test samples: 50 normal traffic samples and 50 attack traffic samples.
This makes our experiment dataset-driven instead of using only manually created input values.

## Firstly, loaded the UNSW-NB15 training and testing files. Then selected six features and scaled them into integers.
Next, trained a small neural model and selected 100 balanced test samples.
For each sample, generated an input file, witness, Groth16 proof, and public output.
Finally, verified all proofs and performed a tampering test.
### used Python for training, Node.js for input generation, Circom for circuit design, Poseidon for hashing, and snarkjs for proof generation and verification.

### To conclude,  AI-based intrusion detection can be made verifiable, privacy-preserving, and tamper-resistant using zero-knowledge proof.
Using UNSW-NB15, our model achieved 81% accuracy on 100 selected samples, and all 100 ZK proofs were successfully verified.
The tampering test produced Invalid proof, confirming output integrity.
