#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>

// ===== TinyML =====
#include "tinyml.h"

// =======================
// HARDWARE CONFIG
// =======================
#define SOUND_SENSOR_PIN 34
#define SERVO_PIN 2

LiquidCrystal_I2C lcd (0x21, 16, 2);
Servo durianServo;

// =======================
// AUDIO FEATURE CONFIG
// =======================
#define SAMPLE_COUNT 400      // ~100ms @4kHz
#define FEATURE_LEN 1         // RMS

float audio_buffer[SAMPLE_COUNT];
float features[FEATURE_LEN];

// =======================
// TINYML GLOBALS (THEO FORM BẠN YÊU CẦU)
// =======================
namespace
{
    tflite::ErrorReporter *error_reporter = nullptr;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    constexpr int kTensorArenaSize = 8 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

// =======================
// TINYML SETUP
// =======================
void setupTinyML()
{
    Serial.println("TensorFlow Lite Init....");

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(sound_classifier_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        error_reporter->Report("Model schema mismatch");
        return;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        error_reporter->Report("AllocateTensors failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("TinyML READY");
}

// =======================
// SERVO
// =======================
void servo_knock()
{
    durianServo.write(90);
    delay(250);
    durianServo.write(0);
    delay(300);
}

// =======================
// AUDIO + FEATURE
// =======================
void collect_audio()
{
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        audio_buffer[i] = analogRead(SOUND_SENSOR_PIN);
        delayMicroseconds(250); // ~4kHz
    }
}

void extract_features()
{
    float sum = 0;
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        float v = audio_buffer[i] - 2048;
        sum += v * v;
    }
    features[0] = sqrt(sum / SAMPLE_COUNT) / 2048.0;
}

// =======================
// ONE INFERENCE
// =======================
bool run_one_inference(float &score_chin)
{
    collect_audio();
    extract_features();

    input->data.f[0] = features[0];

    if (interpreter->Invoke() != kTfLiteOk)
    {
        Serial.println("Invoke failed");
        return false;
    }

    float score_chua_chin = output->data.f[0];
    score_chin = output->data.f[1];

    Serial.print("Chua chin: ");
    Serial.print(score_chua_chin, 3);
    Serial.print(" | Chin: ");
    Serial.println(score_chin, 3);

    return (score_chin > score_chua_chin);
}

// =======================
// TINYML TASK (VOTE 3 LẦN)
// =======================
void tiny_ml_task(void *pvParameters)
{
    setupTinyML();

    while (1)
    {
        int vote_chin = 0;

        for (int i = 0; i < 3; i++)
        {
            lcd.clear();
            lcd.setCursor(0, 0);
            lcd.print("Go lan ");
            lcd.print(i + 1);

            servo_knock();

            lcd.setCursor(0, 1);
            lcd.print("Thu am...");

            float score_chin = 0;
            if (run_one_inference(score_chin))
            {
                vote_chin++;
            }

            delay(500);
        }

        // ===================
        // FINAL DECISION
        // ===================
        lcd.clear();
        lcd.setCursor(0, 0);

        if (vote_chin >= 2)
        {
            lcd.print("SAU RIENG CHIN");
            Serial.println(">>> FINAL: CHIN");
        }
        else
        {
            lcd.print("CHUA CHIN");
            Serial.println(">>> FINAL: CHUA CHIN");
        }

        lcd.setCursor(0, 1);
        lcd.print("Vote chin: ");
        lcd.print(vote_chin);

        delay(4000);
    }
}

// =======================
// ARDUINO SETUP
// =======================
void setup()
{
    Serial.begin(115200);

    lcd.init();
    lcd.backlight();
    lcd.print("Durian AI");

    durianServo.attach(SERVO_PIN);
    durianServo.write(0);

    xTaskCreatePinnedToCore(
        tiny_ml_task,
        "tinyML",
        8192,
        NULL,
        1,
        NULL,
        1
    );
}

void loop()
{
    delay(1000);
}
