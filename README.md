# BRIDGE: Bridging Vision and Language with Cross-Modal Interaction

Official implementation of the **BRIDGE**

## Overview

BRIDGE is a vision-language model that introduces efficient cross-modal interaction layers to bridge pretrained vision and text encoders. Key features:

- **Dual-stream architecture**: Separate vision (ViT) and text (BERT-style) encoders with lightweight cross-modal interaction layers
- **Dual embeddings**: Maintains both cross-modal and unimodal representations for fast retrieval
- **Gated residual connections**: Learnable gates control information flow between modalities
- **Multi-objective training**: Combines ITC (contrastive), MLM (masked language modeling), MIM (masked image modeling), ITM (image-text matching), and cycle consistency losses
- **Staged training curriculum**: Gradually unfreezes encoders from interaction layers to deeper blocks

## Architecture

BRIDGE processes vision and text inputs through separate encoders, then applies bidirectional cross-attention in a shared latent space at selected layers. This design enables:
- Efficient bi-encoder retrieval using unimodal embeddings
- Rich cross-modal understanding via interaction layers
- Flexible adaptation to downstream tasks (VQA, captioning, retrieval, classification)

## Training Stages

1. **Stage A (Stabilize)**: Freeze encoders, train only interaction layers and gates
2. **Stage B (Align)**: Unfreeze top encoder blocks, train with full objective mix
3. **Stage C (Task Tuning)**: Fine-tune for specific downstream tasks

## Repository

This codebase is adapted from the [BLIP](https://github.com/salesforce/BLIP) repository, with modifications for the BRIDGE architecture and training procedure.


## License

See LICENSE file for details.

