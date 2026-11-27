# Training and validation checklist

- [ ] Read the current DataFountain 552 rules and data license/usage terms.
- [ ] Keep competition archives and extracted images outside the public repository.
- [ ] Use Linux/WSL2 or another supported Linux environment for the selected OneFlow stack.
- [ ] Record Python, OS, NVIDIA driver, CUDA, OneFlow, and upstream repository versions.
- [ ] Run the data preparation script and inspect `dataset/audit.json`.
- [ ] Confirm class count and class names in `dataset/classes.json`.
- [ ] Visually inspect bounding boxes and class IDs on a representative image sample.
- [ ] Verify train/validation split has no image overlap.
- [ ] Check upstream `train.py --help` and current install instructions before choosing command-line arguments.
- [ ] Start with a short smoke test and a conservative batch size; confirm GPU memory usage.
- [ ] Record model configuration, image size, batch size, epochs, optimizer, learning rate, and random seed.
- [ ] Evaluate on the held-out validation set and record metrics with the exact evaluation setup.
- [ ] Run inference on all test images and verify output count and JSON schema.
- [ ] Only report metrics and scores that were actually produced and independently checked.
