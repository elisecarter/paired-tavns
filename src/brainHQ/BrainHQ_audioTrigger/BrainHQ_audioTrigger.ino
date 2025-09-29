#include <Arduino.h>
#include <stdint.h>
#include "template_int16.h"
#include "IRQManager.h"
#include <math.h>
#include "IRQManager.h"
#include <pwm.h>
/*
R4_FastADC_TimedScan.ino
This sketch illustrates using the AGT1 timer as an Event Link Controller (ELC) event
 to make repeated ADC scans at timed intervals.  It includes a software trigger to start
 data collection on a rising or falling edge; without it conversion starts at a random
 point in the waveform being read.

by Jack Short, March 6 2025

This is free software, no restrictions or limitations are placed on its use.
This software is provided "as is" without warranty of any kind with regard
to its usablilty or suitablity for any purpose.
*/

// #defines from Susan Parker's file:  susan_ra4m1_minima_register_defines.h
//  at https://github.com/TriodeGirl/RA4M1_Arduino_UNO-R4_Processor_Direct_Register_Addressing_Defines
// section and page numbes are from the RA4M1 hardware manual
//  https://www.renesas.com/us/en/document/mah/renesas-ra4m1-group-users-manual-hardware


// stimulation parameters
PwmOut stim(D11);
float stim_freq = 30; //Hz
float stim_pw = 200e-6; //s
float period = 1/stim_freq;
float duty_perc = stim_pw / period * 100; //percent

// interrupt controller
#define ICUBASE 0x40000000                                            // ICU Base - See 13.2.6 page 233, 32 bits -
#define ICU_IELSR 0x6300                                              // ICU Event Link Setting Register n
#define ICU_IELSR00 ((volatile unsigned int *)(ICUBASE + ICU_IELSR))  // IELSR register 0
#define IRQ_AGT1_AGTI 0x21                                            // AGT1 underflow interrupt
#define IRQ_ADC140_ADI 0x29                                           // ADC14 scan-end interrupt
#define ICU_IELSR_IR 0x10000                                          // Interrupt Status Flag bit (IR, bit 16), in the IELSRx register, ISR must clear
// system clock divider
#define SYSTEM 0x40010000                                             // ICU Base - See 13.2.6 page 233
#define SYSTEM_PRCR ((volatile unsigned short *)(SYSTEM + 0xE3FE))    // Protect Register
#define SYSTEM_SCKDIVCR ((volatile unsigned int *)(SYSTEM + 0xE020))  // System Clock Division Control Register
// Event Link Contrioller registers
#define ELCBASE 0x40040000                                                      // Event Link Controller
#define ELC_ELCR ((volatile unsigned char *)(ELCBASE + 0x1000))                 // Event Link Controller Register
#define ELC_ELSR 0x1010                                                         // Event Link Setting Registers
#define ELC_ELSR08 ((volatile unsigned short *)(ELCBASE + ELC_ELSR + (8 * 4)))  // ELC_AD00 - ADC14A
// Module Stop Control Registers C and D
#define MSTP 0x40040000                                          // Module Registers
#define MSTP_MSTPCRC ((volatile unsigned int *)(MSTP + 0x7004))  // Module Stop Control Register C
#define MSTPC14 14                                               // ELC   - Event Link Controller Module
#define MSTP_MSTPCRD ((volatile unsigned int *)(MSTP + 0x7008))  // Module Stop Control Register D
#define MSTPD2 2                                                 // AGT1   - Asynchronous General Purpose Timer 1 Module
#define MSTPD16 16                                               // ADC140 - 14-Bit A/D Converter Module
// ADC registers
#define ADCBASE 0x40050000                                              // ADC Base
#define ADC140_ADCSR ((volatile unsigned short *)(ADCBASE + 0xC000))    // A/D Control Register
#define ADCSR_ADST 15                                                   // A/D Conversion Start bit
#define ADC140_ADANSA0 ((volatile unsigned short *)(ADCBASE + 0xC004))  // A/D Channel Select Register A0
#define ADC140_ADCER ((volatile unsigned short *)(ADCBASE + 0xC00E))    // A/D Control Extended Register
#define ADC140_ADSTRGR ((volatile unsigned short *)(ADCBASE + 0xC010))  // A/D Conversion Start Trigger Select Register
#define ADC140_ADDR00 ((volatile unsigned short *)(ADCBASE + 0xC020))   // A1 data register
// ====  Asynchronous General Purpose Timer (AGT) =====
#define AGTBASE 0x40084000
#define AGT0_AGT ((volatile unsigned short *)(AGTBASE))  // AGT Counter Register
#define AGT1_AGT ((volatile unsigned short *)(AGTBASE + 0x100))
#define AGT0_AGTCMA ((volatile unsigned short *)(AGTBASE + 0x002))  // AGT Compare Match A Register
#define AGT1_AGTCMA ((volatile unsigned short *)(AGTBASE + 0x102))
#define AGT0_AGTCMB ((volatile unsigned short *)(AGTBASE + 0x004))  // AGT Compare Match B Register
#define AGT1_AGTCMB ((volatile unsigned short *)(AGTBASE + 0x104))
// 8 bit registers
#define AGT0_AGTCR ((volatile unsigned char *)(AGTBASE + 0x008))     // AGT Control Register
#define AGT1_AGTCR ((volatile unsigned char *)(AGTBASE + 0x108))     //
#define AGTCR_TSTART 0                                               // R/W - AGT Count Start; 1: Count starts, 0: Count stops
#define AGTCR_TCSTF 1                                                // R   - AGT Count Status Flag; 1: Count in progress, 0: Count is stopped
#define AGTCR_TSTOP 2                                                // W   - AGT Count Forced Stop; 1: The count is forcibly stopped, 0: Writing 0 is invalid!!!
#define AGT0_AGTMR1 ((volatile unsigned char *)(AGTBASE + 0x009))    // AGT Mode Register 1
#define AGT1_AGTMR1 ((volatile unsigned char *)(AGTBASE + 0x109))    //
#define AGT0_AGTMR2 ((volatile unsigned char *)(AGTBASE + 0x00A))    // AGT Mode Register 2
#define AGT1_AGTMR2 ((volatile unsigned char *)(AGTBASE + 0x10A))    //
#define AGT0_AGTIOC ((volatile unsigned char *)(AGTBASE + 0x00C))    // AGT I/O Control Register
#define AGT1_AGTIOC ((volatile unsigned char *)(AGTBASE + 0x10C))    //
#define AGTIOC_TOE 2                                                 // AGTOn Output Enable
#define AGT0_AGTISR ((volatile unsigned char *)(AGTBASE + 0x00D))    // AGT Event Pin Select Register
#define AGT1_AGTISR ((volatile unsigned char *)(AGTBASE + 0x10D))    //
#define AGT0_AGTCMSR ((volatile unsigned char *)(AGTBASE + 0x00E))   // AGT Compare Match Function Select Register
#define AGT1_AGTCMSR ((volatile unsigned char *)(AGTBASE + 0x10E))   //
#define AGT0_AGTIOSEL ((volatile unsigned char *)(AGTBASE + 0x00F))  // AGT Pin Select Register
#define AGT1_AGTIOSEL ((volatile unsigned char *)(AGTBASE + 0x10F))  //
// end of Parker code

// AGT clock sources
#define AGTSRC_PCLKB 0    // PCLKB (24 MHz)
#define AGTSRC_PCLKB_8 1  // PCLKB / 8 (3 MHz)
#define AGTSRC_PCLKB_2 3  // PCLKB / 2 (12 MHz)
#define AGTSRC_AGTLCLK 4  // divided clock specified by AGTLCLK (LOCO) (32.768 kHz)
#define AGTSRC_AGT0_UF 5  // AGT0 underflow (1 ms)
#define AGTSRC_AGTSCLK 6  // divided clock specified by AGTSCLK (SOSC) (unusable on UNO R4)

#define AGTMR1_TCK 4  // bit shift for loading clock-source bits into AGTMR1
#define AGT_TIMER 0   // AGT mode - timer

#define AGT_AGTCR_TUNDF 0x20  // TUNDEF bit in AGTCR, clear it in ISR to acknowledge underflow interrupt

// AGTLCLK divisor codes
#define DIVCODE_1_1 0    // 1 / 1
#define DIVCODE_1_2 1    // 1 / 2
#define DIVCODE_1_4 2    // 1 / 4
#define DIVCODE_1_8 3    // 1 / 8
#define DIVCODE_1_16 4   // 1 / 16
#define DIVCODE_1_32 5   // 1 / 32
#define DIVCODE_1_64 6   // 1 / 64
#define DIVCODE_1_128 7  // 1 / 128

#define ICU_IELSR_IR 0x10000  // Interrupt Status Flag bit (IR, bit 16), in the IELSRx register, ISR must clear it

#define ELCON 0x80   // enable bit in the Event Link Controller Register (ELCR)
#define ADST 0x8000  // (1 << ADCSR_ADST), start-conversion bit in the ADCSR register

// commands from controller computer program
#define CMDLEN 6            // length of a command sent to the Arduino from controller program
#define ARDCMD_NULL 0       // no command
#define ARDCMD_STARTSCAN 1  // start sampling
#define ARDCMD_STOPSCAN 2   // stop sampling
#define ARDCMD_SETRATE 3    // set the sampling rate
#define ARDCMD_TRIGTYPE 4   // set trigger type for software trigger
#define ARDCMD_TRIGLVL 5    // set trigger level for software trigger
#define ARDCMD_SAMPRATE 6   // set sampling rate for ELC event

//#define _12BITS 1                                           // un-comment this for 12-bit mode

#define BUFF_SIZE 1600  // fills a good portion of a computer screen
// Buffer used to hold recent samples. We keep the behavior of a block buffer
// for compatibility, but also maintain a small circular history and a
// per-sample ready flag so the main loop can emit samples as they arrive.
unsigned short buffer[BUFF_SIZE];     // block readings go here
uint8_t *pBuff8 = (uint8_t *)buffer;  // byte pointer for Serial.write()

// Windowing parameters: process WINDOW_SIZE samples with 50% overlap.
#define WINDOW_SIZE 800
#define HOP (WINDOW_SIZE / 2)  // 50% overlap
static float templateNorm;

// ring buffer that always holds the latest WINDOW_SIZE samples
volatile unsigned short window[WINDOW_SIZE];
volatile unsigned int writeIndex;              // next write position in window
volatile unsigned int samplesCollected;        // total samples stored so far (capped at WINDOW_SIZE)
volatile unsigned int samplesSinceLastWindow;  // samples since last window trigger
// small FIFO to enqueue ready window start indices (writeIndex snapshot)
#define WQ_LEN 8
volatile unsigned int windowQueue[WQ_LEN];
volatile unsigned int wqHead;  // next enqueue position
volatile unsigned int wqTail;  // next dequeue position

volatile bool windowReady;            // legacy flag (kept for compatibility, but queue is used)
volatile unsigned long isrCount = 0;  // incremented in ISR for diagnostics

GenericIrqCfg_t cfgAdcIrq;  // structure defined in IRQManager.h for settng an IRQ

// zero level for input that has a 2.5 volt offset added
#define ZERO_LVL 8192  // half of 2^14

// note:  the above will need adjustment for the analog reference voltage being < 5.0v.

void ADC_ISR();                            // ADC interrupt service routine
void StopScan();                           // end the current scan, may stop AGT1 too
void SetRate(int nRateCode);               // sets ADC scan rate
void SetTrigType(int nTrgTyp);             // set software trigger type - none, rising edge or falling edge
void SetTrigLevel(unsigned char *pData);   // set software trigger level to bracket on rising or falling edge
void SetSampleRate(unsigned char *pData);  // convert 4-byte rate from little to big-endian and set the AGT1 rate
bool SetAGTRate(unsigned long nMicrosec);  // set AGT1 rate
void StartAGT();                           // start AGT1
void StopAGT();                            // stop AGT1

// for converting a long value between
// little-endian (least-significant byte first) and big-endian (most-significant byte first)
// the RA4M1 uses the big-endian format
typedef union {
  unsigned long n;     // long value
  unsigned char c[4];  // bytes to swap
} LongChar;

// software trigger
#define TRIGTYPE_NONE 0     // no trigger, don't wait to start reading
#define TRIGTYPE_RISING 1   // rising-edge trigger:  fires when readings go from below to above the trigger value
#define TRIGTYPE_FALLING 2  // falling-edge trigger:  fires when readings go from above to below the trigger value
// Software trigger variables removed

bool bAGTRunning;  // true if AGT1 is running, false if it has been stopped.

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;
  // install the ADC IRQ
  cfgAdcIrq.irq = FSP_INVALID_VECTOR;             // initialize structure
  cfgAdcIrq.ipl = 12;                             // priority level
  cfgAdcIrq.event = (elc_event_t)IRQ_ADC140_ADI;  // ADC140_ADI interrupt
  IRQManager::getInstance().addGenericInterrupt(cfgAdcIrq,
                                                ADC_ISR);  // attach the ADC ISR
  // set up the ADC
  *MSTP_MSTPCRD &= (0xFFFFFFFF - (0x01 << MSTPD16));  // clear MSTPD16 bit in Module Stop Control Register D to activate the ADC module

  *ADC140_ADCER = 6;  // 14-bit resolution, no self-diagnosis, flush right


  *ADC140_ADANSA0 = 1;      // using analog pin A1
  *ADC140_ADCSR &= 0x1FFF;  // clear ADCS bits (select single/ELC-triggered mode)

  // not using addition or averaging:  ADADS0, ADADS1 and ADADC are left in their reset state, all 0

  // set up the ELC event to start scans
  *ADC140_ADCSR |= 0x0200;    // set TRGE bit to 1 and EXTRG bit to 0; enables the ELC event to start a scan
  *ADC140_ADSTRGR &= 0xC0FF;  // clear ADSTRGR:TRSA bits
  *ADC140_ADSTRGR |= 0x0900;  // TRSA bits = 001001b (9):  respond to event from ELC_AD00

  // set up the ELC module
  *MSTP_MSTPCRC &= (0xFFFFFFFF - (0x01 << MSTPC14));  // clear MSTPC14 bit in Module Stop Control Register C to activate the ELC module
  *ELC_ELSR08 = IRQ_AGT1_AGTI;                        // set AGT1 as the ADC event source
  *ELC_ELCR |= ELCON;                                 // enable the ELC

  // set up AGT1, clear all AGT registers
  *AGT1_AGTCR = 0;                                   // AGT Control Register
  *AGT1_AGTMR1 = 0;                                  // AGT Mode Register 1
  *AGT1_AGTMR2 = 0;                                  // AGT Mode Register 2
  *AGT1_AGTIOC = 0;                                  // AGT I/O Control Register
  *AGT1_AGTISR = 0;                                  // AGT Event Pin Select Register
  *AGT1_AGTCMSR = 0;                                 // AGT Compare Match Function Select Register
  *AGT1_AGTIOSEL = 0;                                // AGT Pin Select Register
  *MSTP_MSTPCRD &= (0xFFFFFFFF - (0x01 << MSTPD2));  // clear MSTPD2 bit in Module Stop Control Register D to activate AGT1
  SetAGTRate(250);                                   // 250 us -> 4kHz
  StartAGT();                                        // start AGT1

  // initialize windowing/ring-buffer bookkeeping
  writeIndex = 0;
  samplesCollected = 0;
  samplesSinceLastWindow = 0;
  windowReady = false;
  wqHead = 0;
  wqTail = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) window[i] = 0;

  // template norm
  long long sumT2 = 0;
  for (unsigned i = 0; i < TEMPLATE_SIZE; i++) {
    long val = template_signal[i];
    sumT2 += (long long)val * val;
    Serial.println(sumT2);
  }
  templateNorm = sqrt((float)sumT2);
  Serial.print("templNorm: ");
  Serial.println(templateNorm);
}

// Define a threshold for the maximum cross-correlation value
#define CORRELATION_THRESHOLD 0.9f  // normalized correlation threshold (range -1.0 to 1.0)

void loop() {
  // diagnostics: print measured ISR rate and queue length once per second
  // static unsigned long lastDiag = 0;
  // unsigned long now = millis();
  // if (now - lastDiag >= 1000) {
  //   noInterrupts();
  //   unsigned long c = isrCount;
  //   isrCount = 0;
  //   unsigned int qlen = (wqHead + WQ_LEN - wqTail) % WQ_LEN;
  //   interrupts();
  //   Serial.print("ISR rate (Hz): ");
  //   Serial.print(c);
  //   Serial.print("  windowQueueLen: ");
  //   Serial.println(qlen);
  //   lastDiag = now;
  // }

  // Dequeue and process any ready windows
  while (true) {
    unsigned int startIdx;
    noInterrupts();
    if (wqHead == wqTail) {
      interrupts();
      break;  // queue empty
    }
    startIdx = windowQueue[wqTail];
    wqTail++;
    if (wqTail >= WQ_LEN) wqTail = 0;
    interrupts();

    // Copy the window starting at startIdx into a local buffer
    unsigned short localWindow[WINDOW_SIZE];
    for (unsigned int i = 0; i < WINDOW_SIZE; i++) {
      unsigned int idx = startIdx + i;
      if (idx >= WINDOW_SIZE) idx -= WINDOW_SIZE;
      localWindow[i] = window[idx] - ZERO_LVL;
    }

    // Slide the template across the window and compute normalized correlation
    float maxCorrelation = 0;
    for (unsigned int offset = 0; offset <= WINDOW_SIZE - TEMPLATE_SIZE; offset++) {
      long dot = 0;
      long sumX2 = 0;

      for (unsigned int i = 0; i < TEMPLATE_SIZE; i++) {
        int16_t x = localWindow[offset + i];
        int16_t t = template_signal[i];

        dot += (long)x * (long)t;
        sumX2 += (long)x * (long)x;
      }

      if (sumX2 != 0) {
        float correlation = (float)dot / (sqrt((float)sumX2) * templateNorm);
        if (correlation > maxCorrelation) {
          maxCorrelation = correlation;
        }
      }
    }

    // Check if the maximum correlation exceeds the threshold
    if (maxCorrelation > CORRELATION_THRESHOLD) {
      stim.begin(stim_freq, duty_perc);
      delay(500);
      stim.end();
    }
    Serial.println(maxCorrelation);
  }
}


void ADC_ISR()
// ISR for ADC140_ADI
{
  *(ICU_IELSR00 + cfgAdcIrq.irq) &= ~(ICU_IELSR_IR);  // reset interrupt controller
  // Read exactly one ADC sample per ISR invocation (one AGT tick).
  unsigned short reading = *ADC140_ADDR00;
  isrCount++;

  // Write the sample into the ring buffer and advance the write index.
  window[writeIndex] = reading;
  writeIndex++;
  if (writeIndex >= WINDOW_SIZE) writeIndex = 0;

  // Track how many samples we've collected (until we fill WINDOW_SIZE).
  if (samplesCollected < WINDOW_SIZE) samplesCollected++;

  // Count samples since the last window trigger; when we reach HOP (400 samples)
  // and the ring contains at least WINDOW_SIZE samples, schedule a window.
  samplesSinceLastWindow++;
  if (samplesCollected >= WINDOW_SIZE && samplesSinceLastWindow >= HOP) {
    samplesSinceLastWindow = 0;
    windowQueue[wqHead] = writeIndex;  // Enqueue the start index of the latest window
    wqHead++;
    if (wqHead >= WQ_LEN) wqHead = 0;
  }
}


void StopScan()
// stop a scan
{
  *ADC140_ADCSR &= 0x7FFF;  // clearing ADST bit stops scanning
}

void SetRate(int nRateCode)
// set the ADC scan rate:
// set the PCKC bits for the Periperal Module Clock C (PLCKC) in the System Clock Division Control Register (SCKDIVCR),
//  as described in the RA4M1 hardware manual section 8.2.1 (page 130)
{
  bool bAGTOn = bAGTRunning;               // will be true if AGT1 is currently running
  StopScan();                              // stop scanning
  if (bAGTRunning)                         // if AGT1 is running
    StopAGT();                             //  no ELC event while changing scan rate
  if (nRateCode >= 0 && nRateCode <= 6) {  // make sure code passed to this function is valid
    *SYSTEM_PRCR = 0xA501;                 // enable writing to the clock registers
    *SYSTEM_SCKDIVCR &= 0xFFFFFF8F;        // zero all PCKC bits
    *SYSTEM_SCKDIVCR |= (nRateCode << 4);  // put in new PCKC value
    *SYSTEM_PRCR = 0xA500;                 // disable writing to the clock registers
  }
  if (bAGTOn)    // if the AGT was running
    StartAGT();  // restart it
}


#define AGT_MAX 0xFFFF  // AGT counters are 16-bit

bool SetAGTRate(unsigned long nMicrosec)
// set rate of AGT1, the sampling rate - how often a n-point scan is made
// the maximum possible period is 255996093 microseconds, a little over 255.99 seconds (4 min 15.99 sec)
// Slower rates need to come from some other timer
{
  bool bEvenMSec = (nMicrosec % 1000) == 0;   // true if period evenly divisible by 1000 (it's an even millisecond)
  unsigned long nNumMSec = nMicrosec / 1000;  // number of whole millisecods in the period value
  unsigned char nSrcCode = AGTSRC_PCLKB;      // starting source code for AGTMR1 - PCLKB
  unsigned char nDivCode = DIVCODE_1_1;       // starting divisor code for AGTMR2 - 1/1
  unsigned long nTicks = nMicrosec * 24;      // number of ticks of 24-MHz PCLKB
  unsigned short nCount;                      // count to be loaded into AGT1
  // if the requested rate is an even millisecond (right 3 decimal digits all 0) in the range of 1 - 65535
  // we can use AGT0 underflow as the clock which is exactly (within the accuracy limits of the LOCO) 1 kHz
  // this gets the most exact match for any rate from 1 ms to 6.5 minutes
  if (bEvenMSec && nNumMSec > 0 && nNumMSec <= AGT_MAX) {  // if the rate is an even milliseond, and is 1 to AGT_MAX milliseconds
    nSrcCode = AGTSRC_AGT0_UF;                             // clock = AGT0 underflow, it is a 1 ms clock
    nCount = nNumMSec;                                     // count is number of milliseconds, range is 1 - 65535 ms
  }
  // not using AGT0; if interval is short enough, use 24 MHz PCLKB as the source clock
  else if (nTicks <= AGT_MAX)                                   // if total ticks is <= 16-bit maximum (65535 / 24 = about 2730 µs or 2.73 ms )
    nCount = nTicks;                                            //  use undivided PCLKB, range 1 µs - 2.7 ms
  else if (nTicks <= AGT_MAX * 2) {                             // else if ticks are < (max * 2)
    nSrcCode = AGTSRC_PCLKB_2;                                  //  use PCLKB / 2 - 12 MHz
    nCount = nTicks / 2;                                        //  get count - range 2.7 - 5.4 ms (can be lower, but lower already taken care of)
  } else if (nTicks <= AGT_MAX * 8) {                           // else if ticks are < (max * 8)
    nSrcCode = AGTSRC_PCLKB_8;                                  //  use PCLKB / 8 - 3 MHz
    nCount = nTicks / 8;                                        //  count, range 5.4 - 21.8 ms
  } else {                                                      // no AGT0 and interval too long for PCLKB
    nSrcCode = AGTSRC_AGTLCLK;                                  // use AGTLCLK which is the LOCO running at 32.768 kHz (32.768 µs / tick)
                                                                // 32768 because that divided by 2 15 times is 1 sec
    nTicks = (unsigned long)(32768.0 * nMicrosec / 1000000.0);  // convert µs to 32768ths sec
    if (nTicks <= AGT_MAX)                                      // if total ticks is <= 16-bit maximum
      nCount = nTicks;                                          //  use undivided AGTLCLK, range 30.5 ms - 1.9 sec
    else if (nTicks <= AGT_MAX * 2) {                           // else if ticks are < (max * 2)
      nDivCode = DIVCODE_1_2;                                   //  use AGTLCLK / 2 - 16.384 kHz
      nCount = nTicks / 2;                                      //  get count, range 1.9 - 3.9 sec
    } else if (nTicks <= AGT_MAX * 4) {                         // else if ticks are < (max * 4)
      nDivCode = DIVCODE_1_4;                                   //  use AGTLCLK / 4 - 8.192 kHz
      nCount = nTicks / 4;                                      //  get count, range 3.9 - 7.9 sec
    } else if (nTicks <= AGT_MAX * 8) {                         // else if ticks are < (max * 8)
      nDivCode = DIVCODE_1_8;                                   //  use AGTLCLK / 8 - 4.096 kHz
      nCount = nTicks / 8;                                      //  get count, range 7.9 - 15.9 sec
    } else if (nTicks <= AGT_MAX * 16) {                        // else if ticks are < (max * 16)
      nDivCode = DIVCODE_1_16;                                  //  use AGTLCLK / 16 - 2.048 kHz
      nCount = nTicks / 16;                                     //  get count, range 15.9 - 31.9 sec
    } else if (nTicks <= AGT_MAX * 32) {                        // else if ticks are < (max * 32)
      nDivCode = DIVCODE_1_32;                                  //  use AGTLCLK / 32 - 1.024 kHz
      nCount = nTicks / 32;                                     //  get count, range 31.9 - 63.9 sec
    } else if (nTicks <= AGT_MAX * 64) {                        // else if ticks are < max * 64
      nDivCode = DIVCODE_1_64;                                  //  use AGTLCLK / 64 - 512 Hz
      nCount = nTicks / 64;                                     //  get count, range 63.9 - 127.9 sec
    } else if (nTicks <= AGT_MAX * 128) {                       // else if ticks are < max * 128
      nDivCode = DIVCODE_1_128;                                 //  use AGTLCLK / 128 - 256 Hz
      nCount = nTicks / 128;                                    //  get count, range 127.9 - 255.9 sec
    } else                                                      // 255996093 microseconds (255.99 sec.) gives a starting count of 65535 for AGTL / 128 , 255998046 overflows to here
      return false;                                             // interval too long for AGT
  }
  *AGT1_AGTMR1 = (nSrcCode << AGTMR1_TCK) | AGT_TIMER;  // timer mode, and click source
  *AGT1_AGTMR2 = nDivCode;                              // divisor code (0 if source is not AGTLCLK)
  *AGT1_AGT = (unsigned short)nCount;                   // count goes in counter reload register
  return true;
}

void StartAGT()
// start AGT1
{
  *AGT1_AGTCR |= 1;    // write 1 to TSTART to start AGT1
  bAGTRunning = true;  // let the rest of the sketch know
}

void StopAGT()
// stop AGT1
{
  *AGT1_AGTCR &= 0xFE;  // clear TSTART bit to stop AGT1
  bAGTRunning = false;  // let the rest of the sketch know
  while ((*AGT1_AGTCR & 2) != 0)
    ;  // wait for TCSTF bit to clear
}