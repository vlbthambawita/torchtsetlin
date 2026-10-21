# Literature for the torchtsetlin JMLR MLOSS paper

Every key below is an entry in [`../paper/references.bib`](../paper/references.bib). Metadata was checked
on 2026-10-21 against arXiv abstract pages, Crossref, PMLR, JMLR, NeurIPS proceedings, GitHub and PyPI.
Where a paper has a peer-reviewed version, the bib entry cites that version and the arXiv id goes in `note`.
PDFs live in [`papers/`](papers/). Each one was copied from `source_documents/papers/` or downloaded from an
open-access source and checked to be a real PDF.

"paywalled" means no legal open-access copy could be fetched. "software" means a repository, so there is no PDF.

## TM foundations

| key | citation (short) | year | relevance to this paper | PDF |
|---|---|---|---|---|
| `tsetlin1961` | Tsetlin, *On behaviour of finite automata in random medium*, Avtomatika i Telemekhanika 22(10) | 1961 | Origin of the Tsetlin automaton, which is the integer state behind `ta_state` | paywalled |
| `narendra1989learning` | Narendra & Thathachar, *Learning Automata: An Introduction*, Prentice Hall | 1989 | Standard reference for learning-automata theory in the background section | paywalled (book) |
| `granmo2018tsetlin` | Granmo, *The Tsetlin Machine: A game theoretic bandit driven approach…*, arXiv:1804.01508 | 2018 | Defines the TM, clauses and Type I/II feedback. `functional.py` implements this algorithm | [pdf](papers/granmo2018tsetlin.pdf) |
| `granmo_book` | Granmo, *An Introduction to Tsetlin Machines*, online book (tsetlinmachine.org) | 2021 | Tutorial reference for readers new to TMs. The book exists as chapters released 2021–2023 | [ch.1](papers/granmo_book.pdf), [ch.2](papers/granmo_book_ch2.pdf), [ch.4](papers/granmo_book_ch4.pdf), [ch.7](papers/granmo_book_ch7.pdf) |

## TM variants implemented or related

| key | citation (short) | year | relevance to this paper | PDF |
|---|---|---|---|---|
| `granmo2019convolutional` | Granmo et al., *The Convolutional Tsetlin Machine*, arXiv:1905.09688 | 2019 | The basis for the `Conv*` models: patches, position encoding, and the random matching patch drawn for feedback | [pdf](papers/granmo2019convolutional.pdf) |
| `phoulady2019weighted` | Phoulady et al., *The Weighted Tsetlin Machine*, arXiv:1911.12607 (StarAI 2020 workshop) | 2019 | Real-valued clause weights, one of the two weighting schemes | [pdf](papers/phoulady2019weighted.pdf) |
| `abeyrathna2021integer` | Abeyrathna, Granmo & Goodwin, *Integer-weighted clauses*, IEEE Access 9 | 2021 | Integer-weighted TM, which tmu cites as "Weighted TM". Cite alongside `phoulady2019weighted` | paywalled (OA at IEEE, not fetched) |
| `abeyrathna2020regression` | Abeyrathna et al., *The Regression Tsetlin Machine*, Phil. Trans. R. Soc. A 378(2164) | 2020 | The `RegressionTsetlinMachine` output layer and feedback policy | not fetched (publisher blocks scripted download) |
| `abeyrathna2021convregression` | Abeyrathna et al., *Convolutional Regression TM*, ICMLT 2021 (ACM) | 2021 | Prior art for convolutional regression | [pdf](papers/abeyrathna2021convregression.pdf) |
| `glimsdal2021coalesced` | Glimsdal & Granmo, *Coalesced Multi-Output TMs with Clause Sharing*, arXiv:2108.07594 | 2021 | The `Coalesced*` models: a shared clause pool with per-class signed weights | [pdf](papers/glimsdal2021coalesced.pdf) |
| `sharma2023dropclause` | Sharma et al., *Drop Clause*, AAAI 37 | 2023 | The clause-dropout masks applied during learning | [pdf](papers/sharma2023dropclause.pdf) (AAAI camera-ready) |
| `abeyrathna2023clausesize` | Abeyrathna et al., *Constraining TM Clause Size*, IJCAI 2023 | 2023 | `max_included_literals`, the clause budget (a hard cap in torchtsetlin) | [pdf](papers/abeyrathna2023clausesize.pdf) (IJCAI) |
| `sparsetm2024` | Østby, Brambo & Glimsdal, *The Sparse TM*, IEEE IC-NIDC 2025 (arXiv:2405.02375) | 2025 | Sparse clause representation, related efficiency work | [pdf](papers/sparsetm2024.pdf) (arXiv) |
| `contracting2023` | Bhattarai et al., *Contracting TM with Absorbing Automata*, ISTM 2023 | 2023 | Sparse/absorbing TA states, related efficiency work | [pdf](papers/contracting2023.pdf) (arXiv) |
| `probabilistictm2024` | Abeyrathna et al., *The Probabilistic TM*, ICAAI 2024 (ACM) | 2024 | Variant not implemented here. Cited for scope | [pdf](papers/probabilistictm2024.pdf) (arXiv) |
| `granmo2025graph` | Granmo et al., *The TM Goes Deep: Logical Learning and Reasoning With Graphs*, arXiv:2507.14874 | 2025 | Graph TM, which is out of scope. Cited when comparing variant coverage | [pdf](papers/granmo2025graph.pdf) |
| `granmo2023composites` | Granmo, *TMComposites*, arXiv:2309.04801 | 2023 | Ensembles of specialised TMs. Related to the multi-model examples | [pdf](papers/granmo2023composites.pdf) |
| `halenka2024hypervectors` | Halenka et al., *Effects of Hyperdimensional Vectors for TMs*, ISTM 2024 | 2024 | Hypervector input encodings, related to the Booleanization encoders | [pdf](papers/halenka2024hypervectors.pdf) (arXiv) |
| `ctmunet2025` | Liu & Abeyrathna, *CTM-UNet*, ISTM 2025 | 2025 | Prior art for TM semantic segmentation (the `segmentation` models) | [pdf](papers/ctmunet2025.pdf) |

## Interpretation, speed and applications

| key | citation (short) | year | relevance to this paper | PDF |
|---|---|---|---|---|
| `blakely2020closedform` | Blakely & Granmo, *Closed-form expressions for global and local interpretation of TMs*, IEA/AIE 2021, LNCS 12798 | 2021 | Feature-importance formulas behind `interpret.py` | [pdf](papers/blakely2020closedform.pdf) (arXiv) |
| `gorji2020indexing` | Gorji et al., *Clause indexing*, IEA/AIE 2020, LNCS 12144 | 2020 | CPU speed-up technique. Contrasts with dense tensor evaluation on the GPU | [pdf](papers/gorji2020indexing.pdf) (arXiv) |
| `abeyrathna2021parallel` | Abeyrathna et al., *Massively parallel and asynchronous TM architecture*, ICML 2021, PMLR 139 | 2021 | The main prior GPU TM (PyTsetlinMachineCUDA). It uses asynchronous per-clause updates, whereas torchtsetlin uses batched feedback | [pdf](papers/abeyrathna2021parallel.pdf) (PMLR) |
| `berge2018medical` | Berge et al., *TM for interpretable text categorization with medical applications*, IEEE Access 7 | 2019 | Medical interpretability use case and the motivating application domain | [pdf](papers/berge2018medical.pdf) (arXiv) |
| `pvc2023` | Zhang et al., *Interpretable TM-based PVC identification*, arXiv:2301.10181 | 2023 | Medical (ECG) application of TMs | [pdf](papers/pvc2023.pdf) |

## Hardware

| key | citation (short) | year | relevance to this paper | PDF |
|---|---|---|---|---|
| `wheeldon2020asynchronous` | Wheeldon et al., *Low-latency asynchronous logic design for inference at the edge*, DATE 2021 (arXiv:2012.03402) | 2021 | TM edge-inference hardware, showing why Boolean models matter for deployment. **The key matches arXiv:2012.03402**, the asynchronous-logic paper | [pdf](papers/wheeldon2020asynchronous.pdf) (arXiv) |
| `wheeldon2020learning` | Wheeldon et al., *Learning automata based energy-efficient AI hardware design for IoT*, Phil. Trans. R. Soc. A 378(2182) | 2020 | The other Wheeldon paper (the first TM ASIC), added so either can be cited | not fetched |
| `impact2024` | Ghazal et al., *IMPACT: in-memory computing for coalesced TM inference*, Phil. Trans. R. Soc. A 383(2288) | 2025 | Hardware target for coalesced TMs | [pdf](papers/impact2024.pdf) (arXiv) |
| `tsetlinkws2025` | Lin et al., *TsetlinKWS: 65nm convolutional TM accelerator for keyword spotting*, arXiv:2510.24282 | 2025 | Recent convolutional TM ASIC | [pdf](papers/tsetlinkws2025.pdf) |
| `maheshwari2023redress` | Maheshwari et al., *REDRESS*, IEEE TPAMI 45(9) | 2023 | Compressing trained TMs for edge microcontrollers, a possible export target | [pdf](papers/maheshwari2023redress.pdf) (Newcastle repository copy) |

## Existing TM software

| key | citation (short) | year | relevance to this paper | PDF |
|---|---|---|---|---|
| `software_pytsetlinmachine` | Granmo, pyTsetlinMachine (C + ctypes), v0.6.6 | 2024 | Reference CPU implementation in the comparison table | software |
| `software_pytsetlinmachinecuda` | Granmo, PyTsetlinMachineCUDA (PyCUDA), v0.1.9 | 2021 | Prior GPU implementation in the comparison table | software |
| `software_tmu` | Granmo & Andersen, TMU, v0.8.3 | 2024 | The most complete current library, with C and CUDA backends. Used as the accuracy baseline | software |
| `software_greentsetlin` | Glimsdal, green-tsetlin (C++/pybind11), v1.1.0 | 2025 | Production-oriented CPU library in the comparison table | software |
| `greentsetlin2024` | Glimsdal et al., *Green Tsetlin: Redefining efficiency in TM frameworks*, arXiv:2405.04212 | 2024 | Paper describing green-tsetlin | [pdf](papers/greentsetlin2024.pdf) |
| `torchtsetlin` | Thambawita, torchtsetlin v0.3.2 | 2026 | The software this paper describes | software |

## Ecosystem and ML context

| key | citation (short) | year | relevance to this paper | PDF |
|---|---|---|---|---|
| `paszke2019pytorch` | Paszke et al., *PyTorch*, NeurIPS 32 | 2019 | Host framework: `nn.Module`, buffers, devices, `state_dict` | [pdf](papers/paszke2019pytorch.pdf) |
| `pedregosa2011scikit` | Pedregosa et al., *Scikit-learn*, JMLR 12 | 2011 | API-design precedent (`fit`/`predict`) and an MLOSS reference point | [pdf](papers/pedregosa2011scikit.pdf) |
| `bottou2018optimization` | Bottou, Curtis & Nocedal, *Optimization methods for large-scale ML*, SIAM Review 60(2) | 2018 | Mini-batch vs. per-example updates, the analogy for batched TM feedback | [pdf](papers/bottou2018optimization.pdf) (arXiv) |
| `rudin2019stop` | Rudin, *Stop explaining black box models…*, Nature Machine Intelligence 1(5) | 2019 | Motivation for inherently interpretable models | [pdf](papers/rudin2019stop.pdf) (arXiv) |
| `cohen1995ripper` | Cohen, *Fast effective rule induction (RIPPER)*, ICML 1995 | 1995 | Classical rule learner, placing TMs among rule-based methods | paywalled |
| `angelino2018corels` | Angelino et al., *Learning certifiably optimal rule lists for categorical data (CORELS)*, JMLR 18 | 2018 | Interpretable rule-list learning, related work | [pdf](papers/angelino2018corels.pdf) |
| `singh2021imodels` | Singh et al., *imodels*, JOSS 6(61) | 2021 | Interpretable-model library that serves as a software comparison point | [pdf](papers/singh2021imodels.pdf) |
| `petersen2022difflogic` | Petersen et al., *Deep differentiable logic gate networks*, NeurIPS 35 | 2022 | Logic-based models trained on the GPU through PyTorch. The closest gradient-based counterpart | [pdf](papers/petersen2022difflogic.pdf) |
| `lecun1998mnist` | LeCun et al., *Gradient-based learning applied to document recognition*, Proc. IEEE 86(11) | 1998 | MNIST benchmark | [pdf](papers/lecun1998mnist.pdf) |
| `xiao2017fashion` | Xiao, Rasul & Vollgraf, *Fashion-MNIST*, arXiv:1708.07747 | 2017 | Fashion-MNIST benchmark | [pdf](papers/xiao2017fashion.pdf) |
| `krizhevsky2009cifar` | Krizhevsky, *Learning multiple layers of features from tiny images*, tech. report, U. Toronto | 2009 | CIFAR-10 benchmark | [pdf](papers/krizhevsky2009cifar.pdf) |

## Candidates not yet cited

1. **FsTsetlin** (fwaris/FsTsetlin, F#, MIT, last push 2024-09). This is the only other TM library built on
   tensor ops from the PyTorch/libtorch stack (through TorchSharp), and it runs on CPU and GPU. It is the
   closest prior art to "TM as tensor ops", so a reviewer may ask about it.
2. **GraphTsetlinMachine** (cair, PyPI `GraphTsetlinMachine` 0.3.3, 2025-04). The current GPU (PyCUDA)
   library from the TM inventors. It belongs in the software table next to `granmo2025graph`.
3. **pyTsetlinMachineParallel** (cair, PyPI 0.2.1, 2021-10). A multi-threaded CPU version of
   pyTsetlinMachine that uses asynchronous parallel clause updates.
4. **Fuzzy-Pattern TM / Tsetlin.jl** (Hnilov, arXiv:2508.08350, local PDF in `01_foundations/`;
   BooBSD/Tsetlin.jl). A fast Julia implementation with recent results on MNIST and CIFAR-10.
5. **MATADOR** (Rahman et al., DATE 2024, doi:10.23919/DATE58400.2024.10546779). Generates SoC designs
   automatically from trained TMs. Together with REDRESS, it covers the "train in Python, deploy to edge" story.
6. **Tunheim et al., ConvCoTM FPGA accelerator** (ISTM 2022, doi:10.1109/ISTM54910.2022.00013). Hardware
   for the convolutional coalesced TM, the model family torchtsetlin implements.
7. **Bhattarai et al., TM Autoencoder / TM Embedding** (arXiv:2301.00709, local PDF). Implemented in tmu
   but not in torchtsetlin, so useful when stating variant coverage honestly.
8. **Abeyrathna et al., Adaptive sparse representation of continuous input for TMs** (Electronics 10(17),
   2021, doi:10.3390/electronics10172107). The thresholding/Booleanization method, relevant to the encoders in `data/`.
9. **tsetlin-rs** (RAprogramm/tsetlin-rs, crate 0.3.4, 2026-01). A recently maintained Rust library with CI,
   tests and docs. Useful to show that the ecosystem is CPU-only outside the cair CUDA code.
10. **Tarasyuk et al., Learning dynamics, pattern recognition capability and interpretability of the TM**
    (Pattern Recognition 174, 2026, doi:10.1016/j.patcog.2025.113028). Recent analysis of TM learning
    dynamics, relevant to the batch-versus-sequential fidelity discussion.
