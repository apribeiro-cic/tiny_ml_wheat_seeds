#include <cstdio>
#include "pico/stdlib.h"

// -------------------------------------------------------------------
// TensorFlow Lite Micro (via pico-tflmicro)
// -------------------------------------------------------------------
// Biblioteca disponível em: git clone https://github.com/raspberrypi/pico-tflmicro.git
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Modelo convertido em array C (gerado via xxd -i no Colab)
#include "wheat_seeds_model.h"

// API em C que será chamada pelo main.c
#include "tflm_wrapper.h"

// -------------------------------------------------------------------
// Objetos estáticos do TFLM
// -------------------------------------------------------------------
namespace {

// tamanho da arena de tensores (ajuste se der erro de memória)
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// logger de erros
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// modelo e intérprete
static const tflite::Model* model = nullptr;

// registrador de operações (número de ops que vamos registrar)
static tflite::MicroMutableOpResolver<4> resolver;

// intérprete e tensores de entrada/saída
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

}  // namespace

// -------------------------------------------------------------------
// Inicializa o modelo TFLM
// -------------------------------------------------------------------
int tflm_init_model(void) {
    // Aponta para o modelo dentro do array wheat_seeds_model
    model = tflite::GetModel(wheat_seeds_model_tflite);
    
    if (model == nullptr) {
        printf("Erro: modelo nulo.\n");
        return -1;
    }

    // Registrar operações (FullyConnected = Dense Layers)
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape(); // O Keras às vezes adiciona reshape implícito

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        nullptr,  
        nullptr,  
        false      
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors falhou.\n");
        return -2;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    if (!input_tensor || !output_tensor) {
        printf("Erro ao obter tensores de entrada/saida.\n");
        return -3;
    }

    printf("TFLM inicializado com sucesso (Wheat Seeds).\n");
    printf("Dimensoes input: ");
    for (int i = 0; i < input_tensor->dims->size; i++) {
        printf("%d ", input_tensor->dims->data[i]);
    }
    printf("\n");

    return 0;
}

// -------------------------------------------------------------------
// Executa uma inferência no modelo Wheat Seeds
// -------------------------------------------------------------------
int tflm_infer(const float in_features[7], float out_scores[3]) {
    if (!interpreter || !input_tensor || !output_tensor) {
        return -1;
    }

    // Copia as 7 entradas para o tensor de entrada
    for (int i = 0; i < 7; i++) {
        input_tensor->data.f[i] = in_features[i];
    }

    // Executa o modelo
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke falhou.\n");
        return -2;
    }

    // Copia as 3 saídas (uma por classe)
    for (int i = 0; i < 3; i++) {
        out_scores[i] = output_tensor->data.f[i];
    }

    return 0;
}