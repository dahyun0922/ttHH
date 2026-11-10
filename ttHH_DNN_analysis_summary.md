# ttHH 분류 DNN 연구 진행 요약

> **범위:** 첫 DNN 구축부터 feature 개선·검증, 전체 MC의 generator-weighted efficiency 계산, 예비 물리 수율 추정까지
> **분석 대상:** CMS Run III 2024 NanoAOD MC, \(\sqrt{s}=13.6\,\mathrm{TeV}\)
> **상태:** 세 MC process에 대한 탐색적 분류 및 효율 평가 완료. 실제 충돌 데이터의 ttHH 신호 측정은 아직 수행하지 않음.

## 1. 연구 목적과 데이터

목표는 **\(t\bar tHH\) (signal)** 이벤트를 **\(HH\), \(t\bar t\) (background)** 이벤트와 구분하는 DNN을 개발하고, 어떤 물리량이 분류 능력을 향상시키는지 확인하는 것이다. 세 sample은 동일한 포괄적 붕괴 채널을 공유하지 않는다.

| Sample | 생성 채널의 특징 | 원본 ROOT 수 | 이벤트 수 |
|---|---|---:|---:|
| ttHH | `TTto2L2Nu-HHto2B2W` | 220 | 9,987,962 |
| HH | `HHto2B2WtoLNu2Q` | 61 | 1,199,991 |
| ttbar | `TTto2L2Nu` | 780 | 470,123,263 |
| **합계** | | **1,061** | **481,311,216** |

원본 NanoAOD로부터 jet/lepton 정보를 보존하는 **object cache**와 이벤트 단위 변수를 보존하는 **feature cache**를 구축했다. 원본과 캐시의 이벤트 개수 및 signed `genWeight` 합계를 대조해 일치함을 확인했다. DNN 학습 및 테스트에는 서로 분리된 ROOT 파일 집합을 사용했으며, 초기 학습은 클래스마다 같은 수의 이벤트를 추출하는 **unweighted, class-balanced** 방식으로 진행했다.

## 2. 첫 번째 DNN: 기본 8-feature 모델

입력 변수는 `nSelectedJet`, `nBJets`, `HT`, `ST`, `PuppiMET_pt`, `leading_jet_pt`, `subleading_jet_pt`, `leading_lepton_pt`의 8개였다. 신경망은 **Dense(64) → Dense(64) → Dense(32) → 3-class softmax** 구조이며, Adam optimizer와 early stopping을 사용했다.

첫 실험에서 전체 테스트 정확도는 **74.87%**, ttHH one-vs-rest ROC-AUC는 **0.9441**이었다. 이어서 permutation importance와 ablation을 통해 `ST`, `HT`, lepton \(p_T\), MET 등 기본적인 이벤트 운동학 변수의 유용성을 조사했다. 특히 `ST`를 제거하면 AUC가 약 **0.9443 → 0.9352**로 낮아졌다. 다만 `ST`와 `HT` 등은 상관되어 있어 permutation importance를 독립적인 물리적 인과 기여도로 해석할 수 없다.

## 3. 재구성 후보에서 All-pairs feature로

처음에는 b-jet pair 중 질량이 Higgs 질량(125 GeV)에 가장 가까운 조합, non-b jet pair 중 W 질량(80.4 GeV)에 가장 가까운 조합을 선택하는 재구성 변수를 시험했다. 분류 성능은 개선됐으나 **실제 Higgs/W에서 유래한 jet pair임을 보장하지 않은 채 목표 질량에 가까운 조합을 선택한다**는 문제가 있었다. 이에 따라 해당 방식을 주력 모델에서 제외했다.

대신 이벤트 안의 **모든 b-jet pair**와 **모든 non-b jet pair**를 대상으로 다음 통계량을 계산했다.

- 각 pair의 invariant mass: **min / mean / max**
- 각 pair의 \(\Delta R\): **min / mean / max**

b-jet pair 6개 + non-b jet pair 6개를 기존 8개 변수에 더해 **총 20-feature All-pairs DNN**을 구성했다. pair가 없는 경우에는 통계량에 `-1`을 사용했다. 이 방법은 특정 Higgs/W 후보를 선택하지 않는다.

### Feature 집합별 비교

| Feature 구성 | 입력 수 | ttHH AUC | 테스트 정확도 |
|---|---:|---:|---:|
| 기본 운동학 변수 | 8 | 0.94428 | 0.74740 |
| 기본 + b-jet pair 통계 | 14 | 0.94998 | 0.77698 |
| **기본 + b-jet/non-b jet 전체 pair 통계** | **20** | **0.95179** | **0.78155** |

추가 ablation에서 질량 통계만 추가한 모델은 AUC **0.95015**, \(\Delta R\) 통계만 추가한 모델은 **0.94997**을 기록했다. Pair 존재 여부만 나타내는 flag의 효과는 작았다. 즉, **단순히 pair가 있다는 사실보다 pair의 운동학적 분포가 분류에 도움이 된다**는 근거를 얻었다.

## 4. 모델의 안정성과 물리적 정보 검증

**Jet multiplicity 통제:** 모델이 단지 jet 개수 차이를 학습하는지 확인하기 위해 `nBJets`, `nNonBJets`가 같은 이벤트끼리 비교했다. 예를 들어 `nBJets=2`, `nNonBJets=2`에서 ttHH vs (HH+ttbar) AUC가 **0.8294 → 0.8690**으로 향상됐다. 이는 같은 jet 개수 안에서도 pair 운동학 정보가 유용하다는 뜻이다. 단, 일부 multiplicity 구간은 background 통계가 적다.

**Multi-seed 검증:** training/validation 샘플링과 모델 초기화를 바꾸는 **5개 seed**를 사용하고 동일한 test set으로 평가했다.

| Seed | 기본 8개 AUC | All-pairs 20개 AUC | 차이 |
|---:|---:|---:|---:|
| 42 | 0.94621 | 0.95297 | +0.00676 |
| 123 | 0.94543 | 0.95245 | +0.00702 |
| 2026 | 0.94606 | 0.95291 | +0.00685 |
| 777 | 0.94578 | 0.95265 | +0.00687 |
| 31415 | 0.94602 | 0.95257 | +0.00655 |

평균 개선량은 **\(\Delta\mathrm{AUC}=0.00681\pm0.00017\)**이며, `±`는 반복 학습 간 **표준편차(run-to-run SD)**다. 5번 모두 개선됐다. 이 값은 물리적 측정 오차나 독립적인 test 표본에 대한 신뢰구간은 아니다.

## 5. DNN score 기반 선택 효율

3-class softmax 중 **ttHH score 하나**를 선택 기준으로 삼았다. MC의 실제 process가 무엇이든 같은 조건 \(s_{\mathrm{ttHH}}>c\)를 적용한다. 먼저 held-out test MC에서 ttHH 보존율 90%, 80%, 70%, 50%에 대응하는 score cut을 정하고 HH/ttbar 통과율을 비교했다.

이후 **전체 481,311,216개 MC 이벤트**에 기존 20-feature 모델을 적용해 signed `genWeight`를 반영한 효율을 계산했다.

\[
\epsilon_k^{w}(c)=\frac{\sum_{i\in k,\,s_i>c}w_i^{\mathrm{gen}}}{\sum_{i\in k}w_i^{\mathrm{gen}}},\qquad R_B^w(c)=1-\epsilon_B^w(c).
\]

| 기존 test 목표 | Score cut | 전체 ttHH 보존율 | 전체 HH 제거율 | 전체 ttbar 제거율 |
|---|---:|---:|---:|---:|
| 90% | 0.267144 | 90.2985% | 90.1753% | 82.7614% |
| **80%** | **0.561638** | **80.4277%** | **96.5015%** | **93.4540%** |
| 70% | 0.765470 | 70.4689% | 98.4258% | 97.2199% |
| 50% | 0.937753 | 50.5852% | 99.6290% | 99.4360% |

전체 MC의 signed-weight 합계는 audit와 일치했다: ttHH **9,987,962**, HH **35,550.6025**, ttbar **37,820,110,815.0929**. HH에는 음수 weight가 약 **5.28%**, ttbar에는 약 **0.405%** 존재한다. 80% 목표 cut에서 단순 이벤트 기준 통과율과 weighted 통과율은 HH **3.5914% vs 3.4985%**, ttbar **6.5540% vs 6.5460%**였다.

**기술적 이슈:** 전체 MC 평가 중 ROOT C++과 Python의 \(\Delta R\) 계산값 차이로 `ttbar/0048.root`의 한 경계 이벤트에서 jet cleaning 결과가 달랐다(C++ 약 0.400000006, Python 약 0.399999826). 해당 이벤트를 명시적으로 예외 처리해 계산을 마쳤다. **최종 분석 전에는 두 selection 구현을 일치시켜야 한다.**

## 6. Cross section 확인과 예비 수율 계산

`Runs` tree는 generator weight의 합을 제공하지만 정규화용 cross section을 직접 제공하지 않았다. CMS McM 공개 API를 통해 세 sample의 2024 production fragment를 확인했으나, request에 나타난 `cross_section=-1`(ttHH) 또는 `1`(HH/ttbar)은 신뢰할 물리 단면적으로 확정할 수 없었다.

Generator 설정에서 **ttHH는 top dilepton(`includeTau`) gridpack과 Higgs `bbWW` 선택**, **HH는 `bbWW→bbℓνqq` 형태의 resonance decay filter**, **ttbar는 `TTto2L2Nu` 전용 POWHEG gridpack**을 사용하는 것을 확인했다. 정확한 생성 단계의 branching fraction·filter normalization은 추가 검증이 필요하다.

외부 이론적 inclusive cross section과 붕괴비를 곱하는 **예시 가정**을 사용해 100 fb\(^{-1}\)에서 cut `0.561638`의 수율을 잠정 계산했다. 추정치는 **ttHH 약 1.83건, HH 약 12.9건, ttbar 약 64만 건**이며, 세 process만 고려한 \(S/B\)는 약 **\(2.85\times10^{-6}\)**이었다. **이는 확정된 CMS MC 정규화값이나 실제 데이터의 신호 순도가 아니다.** 특히 붕괴 채널, generator filter와 cross section 정의가 확정되지 않았고 다른 background가 빠져 있다.

## 7. 현재 결론과 다음 단계

**핵심 성과:** 특정 Higgs/W 후보를 목표 질량에 맞춰 선택하지 않고 **모든 jet pair의 invariant mass와 \(\Delta R\) 분포를 요약해도 ttHH 분류 성능이 안정적으로 개선**됨을 확인했다. Jet multiplicity를 고정한 비교와 5회 재학습에서도 이 경향이 유지됐다. 전체 MC에서 동일한 ttHH score cut을 적용한 signed-weighted efficiency도 확보했다.

**남은 주요 과제:** (1) generator gridpack·필터·붕괴비를 포함한 **sample별 유효 cross section 확정**, (2) C++/Python jet selection 통일, (3) 학습과 겹치지 않는 독립 MC 평가 및 validation 기반 cut 정의, (4) tt+heavy flavor, ttH, ttZ, four-top 등 **추가 background와 실제 분석용 preselection·systematic uncertainty** 도입이다.

> **해석 범위:** 현재 결과는 지정된 세 MC sample의 **탐색적 분류·선택 성능**이다. 실제 pp collision에서 ttHH를 발견하거나 단면적을 측정한 결과는 아니다.
