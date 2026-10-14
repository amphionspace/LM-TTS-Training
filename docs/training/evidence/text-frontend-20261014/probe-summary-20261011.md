# /workspace/workspace/yanglin/UltraEval-Audio/init_model/all16-frozen-conditioning-step-00017500
records=3566 frames=358996
with cross-utterance embedding: 2959

## overall (frame-weighted)
| condition | first CE | residual CE | first acc | EOS CE |
| 自身完整 | 1.1151 | 5.4876 | 0.6430 | 0.4001 |
| 自身裁剪 | 1.5236 | 5.5967 | 0.5736 | 0.3520 |
| 同说话人另一条 | 1.4112 | 5.7560 | 0.5876 | 0.9833 |
| 其他说话人 | 1.9010 | 6.3709 | 0.4944 | 0.8873 |

## inside vs outside the reference interval (first codebook CE)
| condition | inside | outside | inside-outside | before | after | acc in | acc out |
| 自身完整 | 1.0990 | 1.1222 | -0.0232 | 1.0992 | 1.1441 | 0.6477 | 0.6410 |
| 自身裁剪 | 1.1045 | 1.7058 | -0.6013 | 1.7561 | 1.6579 | 0.6456 | 0.5424 |
| 同说话人另一条 | 1.3705 | 1.4290 | -0.0585 | 1.3926 | 1.4638 | 0.5953 | 0.5843 |
| 其他说话人 | 1.8697 | 1.9146 | -0.0448 | 1.8725 | 1.9547 | 0.4991 | 0.4924 |

## residual CE inside vs outside
| condition | inside | outside |
| 自身完整 | 5.5335 | 5.4676 |
| 自身裁剪 | 5.5394 | 5.6216 |
| 同说话人另一条 | 5.7778 | 5.7465 |
| 其他说话人 | 6.4292 | 6.3455 |

## first CE by relative position decile
| condition | 0-10% | 10-20% | 20-30% | 30-40% | 40-50% | 50-60% | 60-70% | 70-80% | 80-90% | 90-100% |
| 自身完整 | 1.070 | 1.137 | 1.115 | 1.101 | 1.089 | 1.095 | 1.059 | 1.105 | 1.158 | 1.220 |
| 自身裁剪 | 2.208 | 1.415 | 1.368 | 1.363 | 1.355 | 1.378 | 1.363 | 1.410 | 1.495 | 1.833 |
| 同说话人另一条 | 1.496 | 1.320 | 1.334 | 1.352 | 1.355 | 1.392 | 1.370 | 1.420 | 1.469 | 1.591 |
| 其他说话人 | 1.989 | 1.804 | 1.826 | 1.842 | 1.861 | 1.900 | 1.856 | 1.898 | 1.956 | 2.064 |

## per dataset: first CE (all frames) and crop inside/outside
| dataset | n | full_self | cross | other | crop in | crop out | reliable spk |
| aishell3 | 88 | 1.134 | 1.553 | 1.883 | 1.243 | 2.778 | yes |
| csemotions | 4 | 1.369 | 1.694 | 1.935 | 1.373 | 1.923 | yes |
| emilia | 300 | 1.199 | 1.578 | 2.221 | 1.178 | 1.763 | no |
| emilia_yodas | 300 | 1.167 | 1.521 | 2.160 | 1.110 | 1.719 | no |
| galgame | 300 | 1.609 | nan | 2.365 | 1.590 | 2.194 | yes |
| genshin_voice | 300 | 1.295 | 1.821 | 2.233 | 1.296 | 1.929 | yes |
| hifitts | 300 | 0.796 | 1.296 | 1.615 | 0.802 | 1.631 | yes |
| hifitts2 | 300 | 0.826 | 1.112 | 1.462 | 0.823 | 1.406 | yes |
| libriheavy | 300 | 0.912 | 1.155 | 1.529 | 0.884 | 1.319 | no |
| libritts_r | 300 | 0.821 | 1.207 | 1.446 | 0.837 | 1.381 | yes |
| ljspeech | 13 | 0.712 | 1.060 | nan | 0.690 | 1.283 | yes |
| mls_sidon | 300 | 0.898 | 1.115 | 1.490 | 0.870 | 1.307 | no |
| starrail_voice | 300 | 1.940 | 2.499 | 3.029 | 1.889 | 2.662 | yes |
| vctk | 88 | 0.818 | 1.126 | 1.282 | 1.034 | 2.826 | yes |
| wenetspeech4tts | 300 | 1.358 | nan | 2.300 | 1.433 | 2.278 | no |
| wutheringwaves | 73 | 1.249 | 1.805 | 2.186 | 1.304 | 1.834 | yes |

## embedding geometry
- 自身完整: norm median 10.55, cos(to full self) median 1.0000
- 自身裁剪: norm median 10.37, cos(to full self) median 0.9895
- 同说话人另一条: norm median 10.56, cos(to full self) median 0.9860
- 其他说话人: norm median 10.53, cos(to full self) median 0.9493

## per-utterance: fraction where condition beats cross_same_speaker (first CE)
- 自身完整 < 同说话人另一条: 0.993 (n=2959)
- 自身裁剪 < 同说话人另一条: 0.436 (n=2959)
- 其他说话人 < 同说话人另一条: 0.131 (n=2946)

---

# /workspace/workspace/yanglin/UltraEval-Audio/init_model/all16-frozen-conditioning-step-00047382
records=3566 frames=358996
with cross-utterance embedding: 2959

## overall (frame-weighted)
| condition | first CE | residual CE | first acc | EOS CE |
| 自身完整 | 0.9123 | 5.1051 | 0.6952 | 0.1512 |
| 自身裁剪 | 1.8460 | 5.3886 | 0.5503 | 1.1978 |
| 同说话人另一条 | 1.6497 | 5.6277 | 0.5601 | 1.7678 |
| 其他说话人 | 2.2460 | 6.3608 | 0.4545 | 1.7337 |

## inside vs outside the reference interval (first codebook CE)
| condition | inside | outside | inside-outside | before | after | acc in | acc out |
| 自身完整 | 0.9041 | 0.9159 | -0.0119 | 0.9053 | 0.9261 | 0.6967 | 0.6945 |
| 自身裁剪 | 0.9982 | 2.2148 | -1.2166 | 2.2168 | 2.2129 | 0.6730 | 0.4969 |
| 同说话人另一条 | 1.5629 | 1.6875 | -0.1247 | 1.6376 | 1.7354 | 0.5711 | 0.5553 |
| 其他说话人 | 2.1565 | 2.2849 | -0.1284 | 2.2301 | 2.3372 | 0.4628 | 0.4508 |

## residual CE inside vs outside
| condition | inside | outside |
| 自身完整 | 5.1520 | 5.0848 |
| 自身裁剪 | 5.1734 | 5.4823 |
| 同说话人另一条 | 5.6284 | 5.6274 |
| 其他说话人 | 6.3982 | 6.3445 |

## first CE by relative position decile
| condition | 0-10% | 10-20% | 20-30% | 30-40% | 40-50% | 50-60% | 60-70% | 70-80% | 80-90% | 90-100% |
| 自身完整 | 0.858 | 0.947 | 0.933 | 0.916 | 0.903 | 0.897 | 0.861 | 0.898 | 0.938 | 0.972 |
| 自身裁剪 | 3.043 | 1.541 | 1.480 | 1.500 | 1.506 | 1.549 | 1.569 | 1.663 | 1.846 | 2.661 |
| 同说话人另一条 | 2.036 | 1.396 | 1.436 | 1.486 | 1.537 | 1.587 | 1.598 | 1.662 | 1.742 | 1.981 |
| 其他说话人 | 2.719 | 1.974 | 2.003 | 2.070 | 2.121 | 2.202 | 2.167 | 2.251 | 2.341 | 2.571 |

## per dataset: first CE (all frames) and crop inside/outside
| dataset | n | full_self | cross | other | crop in | crop out | reliable spk |
| aishell3 | 88 | 0.875 | 2.101 | 2.498 | 1.136 | 4.428 | yes |
| csemotions | 4 | 1.096 | 1.745 | 1.998 | 1.281 | 2.423 | yes |
| emilia | 300 | 0.986 | 1.829 | 2.654 | 1.069 | 2.223 | no |
| emilia_yodas | 300 | 0.999 | 1.808 | 2.660 | 1.024 | 2.234 | no |
| galgame | 300 | 1.177 | nan | 2.517 | 1.293 | 2.619 | yes |
| genshin_voice | 300 | 0.986 | 2.033 | 2.544 | 1.111 | 2.368 | yes |
| hifitts | 300 | 0.655 | 1.905 | 2.346 | 0.781 | 2.621 | yes |
| hifitts2 | 300 | 0.706 | 1.371 | 1.854 | 0.798 | 1.978 | yes |
| libriheavy | 300 | 0.790 | 1.336 | 1.858 | 0.846 | 1.706 | no |
| libritts_r | 300 | 0.708 | 1.674 | 1.951 | 0.819 | 2.014 | yes |
| ljspeech | 13 | 0.591 | 1.379 | nan | 0.716 | 1.990 | yes |
| mls_sidon | 300 | 0.761 | 1.230 | 1.739 | 0.805 | 1.658 | no |
| starrail_voice | 300 | 1.534 | 2.587 | 3.146 | 1.567 | 3.064 | yes |
| vctk | 88 | 0.691 | 1.590 | 1.741 | 1.045 | 4.184 | yes |
| wenetspeech4tts | 300 | 1.065 | nan | 2.538 | 1.258 | 3.001 | no |
| wutheringwaves | 73 | 0.955 | 2.079 | 2.477 | 1.107 | 2.316 | yes |

## embedding geometry
- 自身完整: norm median 10.55, cos(to full self) median 1.0000
- 自身裁剪: norm median 10.37, cos(to full self) median 0.9895
- 同说话人另一条: norm median 10.56, cos(to full self) median 0.9860
- 其他说话人: norm median 10.53, cos(to full self) median 0.9493

## per-utterance: fraction where condition beats cross_same_speaker (first CE)
- 自身完整 < 同说话人另一条: 0.999 (n=2959)
- 自身裁剪 < 同说话人另一条: 0.405 (n=2959)
- 其他说话人 < 同说话人另一条: 0.190 (n=2946)

---

# /workspace/workspace/yanglin/UltraEval-Audio/init_model/all16-textbase-step-00017500
records=3566 frames=358996
with cross-utterance embedding: 2959

## overall (frame-weighted)
| condition | first CE | residual CE | first acc | EOS CE |
| 自身完整 | 1.1457 | 5.5908 | 0.6371 | 0.2961 |
| 自身裁剪 | 1.4848 | 5.6460 | 0.5787 | 0.4315 |
| 同说话人另一条 | 1.3606 | 5.7296 | 0.5981 | 0.9120 |
| 其他说话人 | 1.7899 | 6.1205 | 0.5141 | 0.8867 |

## inside vs outside the reference interval (first codebook CE)
| condition | inside | outside | inside-outside | before | after | acc in | acc out |
| 自身完整 | 1.1280 | 1.1534 | -0.0254 | 1.1333 | 1.1725 | 0.6409 | 0.6355 |
| 自身裁剪 | 1.1519 | 1.6296 | -0.4778 | 1.6315 | 1.6279 | 0.6358 | 0.5539 |
| 同说话人另一条 | 1.3221 | 1.3774 | -0.0553 | 1.3311 | 1.4218 | 0.6065 | 0.5945 |
| 其他说话人 | 1.7572 | 1.8042 | -0.0470 | 1.7482 | 1.8575 | 0.5211 | 0.5111 |

## residual CE inside vs outside
| condition | inside | outside |
| 自身完整 | 5.6381 | 5.5703 |
| 自身裁剪 | 5.6455 | 5.6462 |
| 同说话人另一条 | 5.7526 | 5.7195 |
| 其他说话人 | 6.1722 | 6.0980 |

## first CE by relative position decile
| condition | 0-10% | 10-20% | 20-30% | 30-40% | 40-50% | 50-60% | 60-70% | 70-80% | 80-90% | 90-100% |
| 自身完整 | 1.108 | 1.175 | 1.153 | 1.129 | 1.116 | 1.123 | 1.092 | 1.135 | 1.183 | 1.239 |
| 自身裁剪 | 1.935 | 1.382 | 1.351 | 1.356 | 1.349 | 1.370 | 1.354 | 1.392 | 1.454 | 1.863 |
| 同说话人另一条 | 1.397 | 1.280 | 1.288 | 1.304 | 1.309 | 1.349 | 1.333 | 1.366 | 1.404 | 1.563 |
| 其他说话人 | 1.824 | 1.695 | 1.720 | 1.735 | 1.748 | 1.804 | 1.763 | 1.780 | 1.830 | 1.987 |

## per dataset: first CE (all frames) and crop inside/outside
| dataset | n | full_self | cross | other | crop in | crop out | reliable spk |
| aishell3 | 88 | 1.267 | 1.493 | 1.745 | 1.388 | 2.772 | yes |
| csemotions | 4 | 1.427 | 1.626 | 2.001 | 1.549 | 1.792 | yes |
| emilia | 300 | 1.236 | 1.539 | 2.105 | 1.224 | 1.662 | no |
| emilia_yodas | 300 | 1.194 | 1.474 | 2.068 | 1.145 | 1.616 | no |
| galgame | 300 | 1.635 | nan | 2.250 | 1.673 | 2.153 | yes |
| genshin_voice | 300 | 1.337 | 1.737 | 2.084 | 1.362 | 1.830 | yes |
| hifitts | 300 | 0.823 | 1.193 | 1.476 | 0.826 | 1.483 | yes |
| hifitts2 | 300 | 0.842 | 1.094 | 1.392 | 0.858 | 1.384 | yes |
| libriheavy | 300 | 0.922 | 1.119 | 1.442 | 0.915 | 1.308 | no |
| libritts_r | 300 | 0.840 | 1.150 | 1.351 | 0.876 | 1.235 | yes |
| ljspeech | 13 | 0.756 | 1.045 | nan | 0.756 | 1.272 | yes |
| mls_sidon | 300 | 0.906 | 1.072 | 1.377 | 0.886 | 1.250 | no |
| starrail_voice | 300 | 2.012 | 2.414 | 2.880 | 2.014 | 2.544 | yes |
| vctk | 88 | 0.874 | 1.031 | 1.173 | 1.096 | 2.838 | yes |
| wenetspeech4tts | 300 | 1.446 | nan | 2.135 | 1.499 | 2.110 | no |
| wutheringwaves | 73 | 1.297 | 1.721 | 2.085 | 1.358 | 1.718 | yes |

## embedding geometry
- 自身完整: norm median 10.55, cos(to full self) median 1.0000
- 自身裁剪: norm median 10.37, cos(to full self) median 0.9895
- 同说话人另一条: norm median 10.56, cos(to full self) median 0.9860
- 其他说话人: norm median 10.53, cos(to full self) median 0.9493

## per-utterance: fraction where condition beats cross_same_speaker (first CE)
- 自身完整 < 同说话人另一条: 0.975 (n=2959)
- 自身裁剪 < 同说话人另一条: 0.425 (n=2959)
- 其他说话人 < 同说话人另一条: 0.144 (n=2946)

---

