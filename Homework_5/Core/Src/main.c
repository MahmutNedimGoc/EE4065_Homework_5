/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"
#include "mfcc_data.h"
#include <math.h>
#include "mfcc_data.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
ai_u8 activation_data[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
ai_float ai_in_data[AI_NETWORK_IN_1_SIZE];
ai_float ai_out_data[AI_NETWORK_OUT_1_SIZE];
ai_handle network = AI_HANDLE_NULL;
/* USER CODE END Includes */
int global_sonuc = -1;
int global_sayac = 0;
/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
float received_audio[6144]; // 6 parça ses bekliyoruz
float real_buf[1024]; // FFT Reel Kısmı
float imag_buf[1024]; // FFT Sanal Kısmı
float mel_buf[20];
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

// FFT Handler

void pure_fft(float *real, float *imag, int n) {
    // 1. Bit-reversal Permutation
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (j > i) {
            // Swap Real
            float temp = real[j]; real[j] = real[i]; real[i] = temp;
            // Swap Imag
            temp = imag[j]; imag[j] = imag[i]; imag[i] = temp;
        }
        int m = n / 2;
        while (m >= 1 && j >= m) {
            j -= m;
            m /= 2;
        }
        j += m;
    }

    // 2. Danielson-Lanczos (Butterfly)
    for (int m = 1; m < n; m *= 2) { // Adım boyutu
        float wm_r = cosf(M_PI / m); // W_m Real (Euler)
        float wm_i = -sinf(M_PI / m); // W_m Imag (Euler)

        // Trigonometrik yineleme için
        float w_r = 1.0f;
        float w_i = 0.0f;

        for (int k = 0; k < m; k++) { // Kelebek işlemleri
            for (int i = k; i < n; i += 2 * m) {
                int j = i + m;
                // Karmaşık çarpma: t = w * data[j]
                float t_r = w_r * real[j] - w_i * imag[j];
                float t_i = w_r * imag[j] + w_i * real[j];

                // Kelebek toplama/çıkarma
                float u_r = real[i];
                float u_i = imag[i];

                real[i] = u_r + t_r;
                imag[i] = u_i + t_i;
                real[j] = u_r - t_r;
                imag[j] = u_i - t_i;
            }
            // W değerini güncelle: w = w * wm
            float temp_r = w_r * wm_r - w_i * wm_i;
            w_i = w_r * wm_i + w_i * wm_r;
            w_r = temp_r;
        }
    }
}

// --- 2. MFCC ÇIKARMA FONKSİYONU ---
void Extract_Features_Pure(float* input_audio, float* mfcc_output) {

    // A. Pre-emphasis & Windowing
    float prev = input_audio[0] * 32768.0f;

    for (int i = 0; i < 1024; i++) {
        float current = input_audio[i] * 32768.0f;

        // Pre-emphasis
        float val = (i > 0) ? (current - 0.97f * prev) : current;
        prev = current;

        // Windowing
        real_buf[i] = val * hamming_window[i];
        imag_buf[i] = 0.0f;
    }

    // B. FFT
    pure_fft(real_buf, imag_buf, 1024);

    // C. Power Spectrum (|Mag|^2 / N)
    for(int i=0; i < 513; i++) {
        float mag_sq = (real_buf[i] * real_buf[i]) + (imag_buf[i] * imag_buf[i]);
        real_buf[i] = mag_sq / 512.0f;
    }

    // D. Mel Filterbank
    for (int i = 0; i < 20; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 513; j++) {
            sum += mel_filters[i * 513 + j] * real_buf[j];
        }
        if (sum < 1e-6f) sum = 1e-6f;
        mel_buf[i] = 20.0f * log10f(sum);
    }

    // E. DCT (Matris Çarpımı)
    for (int i = 0; i < 13; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 20; j++) {
            // Header dosyasından gelen dct_matrix
            sum += dct_matrix[i * 20 + j] * mel_buf[j];
        }
        mfcc_output[i] = sum;
    }
}
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_CRC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_CRC_Init();
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */
  ai_error err;
    ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    ai_network_params params = {
        .params = AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        .activations = AI_NETWORK_DATA_ACTIVATIONS(activation_data)
    };
    ai_network_init(network, &params);

  /* USER CODE BEGIN WHILE */
    while (1)
      {

        if (HAL_UART_Receive(&huart2, (uint8_t*)received_audio, 6144 * sizeof(float), 5000) == HAL_OK)
        {
            float avg_mfcc[13] = {0.0f};
            int num_chunks = 6;

            for(int c=0; c<num_chunks; c++) {
                float temp_mfcc[13];
                Extract_Features_Pure(&received_audio[c*1024], temp_mfcc);
                for(int k=0; k<13; k++) avg_mfcc[k] += temp_mfcc[k];
            }

            for(int k=0; k<13; k++) ai_in_data[k] = avg_mfcc[k] / (float)num_chunks;

            ai_i32 n_batch;
            ai_buffer* net_inputs = ai_network_inputs_get(network, &n_batch);
            ai_buffer* net_outputs = ai_network_outputs_get(network, &n_batch);

            ai_buffer ai_input_buf = net_inputs[0];
            ai_buffer ai_output_buf = net_outputs[0];

            ai_input_buf.data = AI_HANDLE_PTR(ai_in_data);
            ai_output_buf.data = AI_HANDLE_PTR(ai_out_data);

            ai_i32 batch = ai_network_run(network, &ai_input_buf, &ai_output_buf);

            if (batch != 1) {
                 global_sonuc = -2; // Hata
            } else {
                // Sonucu Bul (Argmax)
                float max_val = -100.0f;
                int max_idx = -1;
                for(int i=0; i < AI_NETWORK_OUT_1_SIZE; i++) {
                    if (ai_out_data[i] > max_val) {
                        max_val = ai_out_data[i];
                        max_idx = i;
                    }
                }
                global_sonuc = max_idx;

                if(global_sayac < 1000) global_sayac = 1000;
                global_sayac++;
            }
        }
        /* USER CODE END WHILE */
      }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
