# Clean-Lingual 

**Clean-Lingual** is an AI-powered text detoxification project designed to convert malicious or toxic Korean sentences into polite and respectful speech. 

Utilizing a Transformer-based Neural Machine Translation (NMT) architecture, the model learns to understand the nuances of aggressive discourse and provides constructive, polite alternatives. The goal is to promote healthier online communication by mitigating toxic speech through advanced NLP techniques.

## Key Features

*   **Transformer-based Seq2Seq**: Built with OpenNMT-py for robust sequence-to-sequence transformation.
*   **Context-aware Detoxification**: Specializes in converting toxic slang and aggressive comments into professional or polite Korean.
*   **Custom Data Augmentation**: Powered by a curated dataset of over 1,000 high-quality parallel text pairs.
*   **Subword-level Learning**: Utilizes SentencePiece tokenization to handle complex Korean morphological variations and improve generalization.

## Current Status

- [x] **Data Collection**: Sourcing raw toxic text from UnSmile and other datasets.
- [x] **Data Augmentation**: Completed **1,000+ parallel text pairs** (Toxic ↔ Polite).
- [ ] **Tokenizer Training**: Training a custom SentencePiece model for Korean subword processing.
- [ ] **Model Refinement**: Re-training the Transformer model to resolve the "Mode Collapse" (repetitive output) issue.
- [ ] **Inference Engine**: Building a lightweight inference engine for local deployment.

## Technical Stack

- **Framework**: OpenNMT-py
- **Tokenization**: SentencePiece
- **Language**: Python 3.x
- **Infrastructure**: Google Colab (T4 GPU) for training

## 📝 Roadmap

1.  **Phase 1**: Initial baseline training with small dataset (Completed).
2.  **Phase 2**: Bulk data augmentation and diversity enhancement (Completed - 1,000 pairs).
3.  **Phase 3**: Integration of Subword Tokenization and hyperparameter tuning (In Progress).
4.  **Phase 4**: Performance validation and edge-case handling.
5.  **Phase 5**: Deployment as a lightweight API/Library.
