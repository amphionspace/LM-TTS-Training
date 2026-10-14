# /workspace/workspace/yanglin/UltraEval-Audio/init_model/all16-textbase-step-00025000
records=3566 frames=358996
with cross-utterance embedding: 2959

## overall (frame-weighted)
| condition | first CE | residual CE | first acc | EOS CE |
| 自身完整 | 1.0788 | 5.4413 | 0.6520 | 0.5956 |
| 自身裁剪 | 1.4856 | 5.5330 | 0.5826 | 0.6789 |
| 同说话人另一条 | 1.3579 | 5.6443 | 0.5993 | 1.3971 |
| 其他说话人 | 1.8269 | 6.1552 | 0.5093 | 1.3885 |

## inside vs outside the reference interval (first codebook CE)
| condition | inside | outside | inside-outside | before | after | acc in | acc out |
| 自身完整 | 1.0657 | 1.0844 | -0.0187 | 1.0686 | 1.0995 | 0.6541 | 0.6510 |
| 自身裁剪 | 1.0964 | 1.6548 | -0.5585 | 1.6577 | 1.6521 | 0.6465 | 0.5548 |
| 同说话人另一条 | 1.3204 | 1.3743 | -0.0539 | 1.3276 | 1.4191 | 0.6068 | 0.5960 |
| 其他说话人 | 1.7904 | 1.8429 | -0.0525 | 1.7800 | 1.9028 | 0.5153 | 0.5067 |

## residual CE inside vs outside
| condition | inside | outside |
| 自身完整 | 5.4853 | 5.4222 |
| 自身裁剪 | 5.5033 | 5.5460 |
| 同说话人另一条 | 5.6646 | 5.6354 |
| 其他说话人 | 6.2057 | 6.1332 |

## first CE by relative position decile
| condition | 0-10% | 10-20% | 20-30% | 30-40% | 40-50% | 50-60% | 60-70% | 70-80% | 80-90% | 90-100% |
| 自身完整 | 1.044 | 1.106 | 1.088 | 1.065 | 1.058 | 1.059 | 1.027 | 1.062 | 1.107 | 1.169 |
| 自身裁剪 | 2.047 | 1.348 | 1.323 | 1.330 | 1.333 | 1.350 | 1.339 | 1.381 | 1.463 | 1.894 |
| 同说话人另一条 | 1.422 | 1.255 | 1.274 | 1.293 | 1.307 | 1.345 | 1.336 | 1.367 | 1.416 | 1.552 |
| 其他说话人 | 1.909 | 1.695 | 1.724 | 1.747 | 1.779 | 1.830 | 1.794 | 1.828 | 1.892 | 2.054 |

## per dataset: first CE (all frames) and crop inside/outside
| dataset | n | full_self | cross | other | crop in | crop out | reliable spk |
| aishell3 | 88 | 1.078 | 1.409 | 1.687 | 1.243 | 2.804 | yes |
| csemotions | 4 | 1.284 | 1.554 | 1.948 | 1.351 | 1.697 | yes |
| emilia | 300 | 1.133 | 1.517 | 2.173 | 1.135 | 1.675 | no |
| emilia_yodas | 300 | 1.156 | 1.507 | 2.219 | 1.118 | 1.685 | no |
| galgame | 300 | 1.502 | nan | 2.135 | 1.513 | 2.056 | yes |
| genshin_voice | 300 | 1.187 | 1.609 | 1.975 | 1.221 | 1.761 | yes |
| hifitts | 300 | 0.797 | 1.258 | 1.607 | 0.846 | 1.625 | yes |
| hifitts2 | 300 | 0.834 | 1.128 | 1.480 | 0.865 | 1.459 | yes |
| libriheavy | 300 | 0.908 | 1.141 | 1.529 | 0.909 | 1.337 | no |
| libritts_r | 300 | 0.833 | 1.231 | 1.460 | 0.883 | 1.365 | yes |
| ljspeech | 13 | 0.733 | 1.075 | nan | 0.769 | 1.362 | yes |
| mls_sidon | 300 | 0.895 | 1.088 | 1.429 | 0.893 | 1.303 | no |
| starrail_voice | 300 | 1.822 | 2.279 | 2.775 | 1.817 | 2.460 | yes |
| vctk | 88 | 0.840 | 1.059 | 1.220 | 1.094 | 2.846 | yes |
| wenetspeech4tts | 300 | 1.259 | nan | 2.089 | 1.338 | 2.113 | no |
| wutheringwaves | 73 | 1.152 | 1.619 | 2.003 | 1.221 | 1.661 | yes |

## embedding geometry
- 自身完整: norm median 10.55, cos(to full self) median 1.0000
- 自身裁剪: norm median 10.37, cos(to full self) median 0.9895
- 同说话人另一条: norm median 10.56, cos(to full self) median 0.9860
- 其他说话人: norm median 10.53, cos(to full self) median 0.9493

## per-utterance: fraction where condition beats cross_same_speaker (first CE)
- 自身完整 < 同说话人另一条: 0.987 (n=2959)
- 自身裁剪 < 同说话人另一条: 0.413 (n=2959)
- 其他说话人 < 同说话人另一条: 0.137 (n=2946)

---

