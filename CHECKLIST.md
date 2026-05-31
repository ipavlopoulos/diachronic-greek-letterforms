# Optical Greek Letters Checklist

Camera-ready and repository tasks after rebuttal.

## Paper Text

- [x] Correct the PaLit-Char date range in the abstract to 2nd-5th century CE.
- [x] Clarify that the project targets robust character representations; diachronic evaluation is the test case.
- [x] Add a short motivation paragraph for LF and DSCL as domain-aware improvements.
- [x] Explain that LF addresses realistic manuscript damage, while DSCL addresses morphologically confusable letter classes.
- [x] Add the standard SCL ablation to the relevant table.
- [x] Report that standard SCL without LF degrades ResNet18-PT+FT from 0.80 to 0.77, while LF+DSCL reaches 0.83.
- [x] Clarify why ResNet18 pretraining alone performs poorly: ImageNet-style photos are far from isolated historical character images.
- [x] Clarify that Hell-Char is derived from Hell-Date for character-level tasks, while PaLit-Char and Med-Char are newly compiled datasets.
- [x] Add technical details for the dynamic similarity matrix:
  - [x] computed over the full training set;
  - [x] class prototypes are built from normalized embeddings;
  - [x] cosine similarities are clamped to [0, 1];
  - [x] diagonal entries are set to zero;
  - [x] matrix is updated every 3 epochs;
  - [x] optional EMA smoothing can be used.
- [x] Add technical details for lacuna augmentation:
  - [x] 1-4 irregular lacunae;
  - [x] each covering 2-15% of image area;
  - [x] ellipse-like masks with random distortion via erosion/dilation;
  - [x] motivated by non-rectangular manuscript damage.
- [x] Add the ViT and ConvNeXt-V2 results, if space permits.
- [x] Mention that LF+DSCL also improves ViT and ConvNeXt-V2 classification performance.
- [x] Add the PaLit-to-Med fine-tuning result, if retained: Med-Char accuracy improves only modestly, from 0.45 to 0.48.
- [x] Explain why PaLit-to-Med transfer remains hard: Roman/late antique majuscule and medieval minuscule are structurally different, and intermediate cursive stages are absent.

## Figures

- [ ] Improve Figure 4 readability.
- [ ] Remove or reduce overlapping captions in Figure 4.
- [ ] Consider drawing polygons or external labels instead of captions over samples.
- [ ] Scale samples more consistently where possible.
- [ ] Add a side-by-side augmentation figure:
  - [ ] standard geometric erasure;
  - [ ] LF lacuna-style erasure;
  - [ ] real damaged character from the dataset, if available.
- [ ] Add a compact visual example of commonly confused characters, e.g. Alpha-Lambda vs. Alpha-Phi.
- [ ] Add a small cross-dataset visual panel, e.g. Gamma examples from Hell-Char, PaLit-Char, and Med-Char.
- [ ] Include enough character examples to show evolution, backgrounds, degradation, and visual confusability.

## Repository

- [x] Rename README heading to Optical Greek Letters.
- [x] Fix nested project-path wording in README.
- [x] Fix inference notebook paths for nested data folders.
- [x] Fix inference notebook prediction variable.
- [x] Add missing `torchvision.models` import in `source.py`.
- [x] Remove duplicated notebook-export definitions from `source.py`.
- [x] Fix TTA label repetition with `repeat_interleave`.
- [x] Fix SupCon self-positive masking.
- [ ] Ensure repository link does not expire.
- [ ] Publish final code and data.
- [ ] Add or link confusion matrix outputs.
- [ ] Add images showing training/inference stages.
- [ ] Add an HTML version of Figure 4 or the Med-Char clustering visualization.
- [ ] Add a short README section pointing reviewers/readers to the visual artifacts.
- [ ] Add reproducibility notes for running the main notebooks.

## Final Sanity Checks

- [ ] Verify all notebook paths work from the `diachronic-greek-letterforms/` directory.
- [ ] Verify saved model inference works on at least one PaLit-Char and one Med-Char sample.
- [ ] Re-run table-generating notebooks or document where table values came from.
- [ ] Confirm all cited table values match the final manuscript.
- [ ] Confirm all dataset date ranges match across abstract, data section, captions, and README.
- [ ] Check that all reviewer-promised changes are visible in either the paper or repository.
