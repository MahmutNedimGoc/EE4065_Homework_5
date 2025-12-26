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
#include <math.h>
#include <string.h> // memset icin
#include "model_data_q2.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// Hu Moment Hesaplama (Ayni fonksiyonu buraya da koyun)
void calculate_hu_moments(const uint8_t img[28][28], float hu[7]) {
    // 1. Raw Moments (m_pq)
    double m00 = 0, m10 = 0, m01 = 0;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            double val = (img[y][x] > 0) ? 1.0 : 0.0;
            m00 += val;
            m10 += x * val;
            m01 += y * val;
        }
    }

    // Eger resim bombossa hata vermesin diye cikis
    if (m00 == 0) {
        for(int i=0; i<7; i++) hu[i] = 0.0f;
        return;
    }

    // 2. Agirlik Merkezi (Centroid)
    double cx = m10 / m00;
    double cy = m01 / m00;

    // 3. Central Moments (mu_pq)
    double mu20 = 0, mu11 = 0, mu02 = 0;
    double mu30 = 0, mu21 = 0, mu12 = 0, mu03 = 0;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            double val = (img[y][x] > 0) ? 1.0 : 0.0;
            double dx = x - cx;
            double dy = y - cy;

            mu20 += dx * dx * val;
            mu11 += dx * dy * val;
            mu02 += dy * dy * val;
            mu30 += dx * dx * dx * val;
            mu21 += dx * dx * dy * val;
            mu12 += dx * dy * dy * val;
            mu03 += dy * dy * dy * val;
        }
    }

    // 4. Normalized Central Moments (nu_pq)
    // Scale invariant olmasi icin
    double inv_m00_2 = 1.0 / (pow(m00, 2));
    double inv_m00_2_5 = 1.0 / (pow(m00, 2.5));

    double nu20 = mu20 * inv_m00_2;
    double nu11 = mu11 * inv_m00_2;
    double nu02 = mu02 * inv_m00_2;
    double nu30 = mu30 * inv_m00_2_5;
    double nu21 = mu21 * inv_m00_2_5;
    double nu12 = mu12 * inv_m00_2_5;
    double nu03 = mu03 * inv_m00_2_5;

    // 5. Hu Moments (h1..h7) - ASIL HESAPLAMA BURASI
    double t1 = nu20 + nu02;
    double t2 = nu20 - nu02;
    double t3 = nu30 - 3 * nu12;
    double t4 = 3 * nu21 - nu03;
    double t5 = nu30 + nu12;
    double t6 = nu21 + nu03;

    hu[0] = (float)t1;
    hu[1] = (float)(t2 * t2 + 4 * nu11 * nu11);
    hu[2] = (float)(t3 * t3 + t4 * t4);
    hu[3] = (float)(t5 * t5 + t6 * t6);
    hu[4] = (float)(t3 * t5 * (t5 * t5 - 3 * t6 * t6) + t4 * t6 * (3 * t5 * t5 - t6 * t6));
    hu[5] = (float)(t2 * (t5 * t5 - t6 * t6) + 4 * nu11 * t5 * t6);
    hu[6] = (float)(t4 * t5 * (t5 * t5 - 3 * t6 * t6) - t3 * t6 * (3 * t5 * t5 - t6 * t6));
}

// MLP Tahmin Fonksiyonu (Ayni kalabilir)
int predict_digit_mlp(float input_hu[7]) {
    // Katman 1
    float hidden1[100];
    for (int i = 0; i < 100; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 7; j++) {
            sum += input_hu[j] * W1[j * 100 + i];
        }
        sum += B1[i];
        hidden1[i] = relu(sum);
    }

    // Katman 2
    float hidden2[100];
    for (int i = 0; i < 100; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) {
            sum += hidden1[j] * W2[j * 100 + i];
        }
        sum += B2[i];
        hidden2[i] = relu(sum);
    }

    // Cikis Katmani
    float output[10];
    for (int i = 0; i < 10; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) {
            sum += hidden2[j] * W3[j * 10 + i];
        }
        sum += B3[i];
        output[i] = sum;
    }

    // Argmax
    int best_class = 0;
    float max_val = output[0];
    for(int i=1; i<10; i++) {
        if(output[i] > max_val) {
            max_val = output[i];
            best_class = i;
        }
    }
    return best_class;
}

/* USER CODE END 0 */

/* USER CODE END 0 */


/* USER CODE END 2 */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */
volatile int result = 0;
volatile float my_hu[7];
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

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  calculate_hu_moments(sample_image, my_hu);
  result = predict_digit_mlp(my_hu);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
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
#ifdef USE_FULL_ASSERT
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
