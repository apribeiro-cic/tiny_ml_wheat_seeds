#ifndef TFLM_WRAPPER_H_
#define TFLM_WRAPPER_H_

#ifdef __cplusplus
extern "C" {
#endif

// Inicializa o modelo (aloca tensores etc.)
// Retorna 0 em sucesso, <0 em erro.
int tflm_init_model(void);

// Executa uma inferência.
// in_features: 7 entradas do Wheat Seeds (Area, Perimeter, Compactness, Length, Width, Asymmetry, Groove)
// out_scores: 3 saídas (probabilidades para Kama, Rosa, Canadian)
// Retorna 0 em sucesso, <0 em erro.
int tflm_infer(const float in_features[7], float out_scores[3]);

#ifdef __cplusplus
}
#endif

#endif  // TFLM_WRAPPER_H_