
/*
  IMU Classifier

  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.

  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry

  This example code is in the public domain.
*/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <ArduinoBLE.h>

#include "model.h"

const float modelThreshold = 0.7;
const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char *GESTURES[] = {
    "stop_right",
    "move_ahead_right",
    "move_back_right"
    };


enum {
  GESTURE_NONE = -1,
  GESTURE_STOP = 0,
  GESTURE_AHEAD = 1,
  GESTURE_BACK = 2
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))
int gesture_left = -1;
int gesture_right = -1;

const char* deviceServiceUuid = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

// CREATE SERVICE AND CHARACTERISTIC
BLEService gestureService(deviceServiceUuid);
BLEByteCharacteristic gestureCharacteristic(deviceServiceCharacteristicUuid, BLERead | BLEWrite);




void setup()
{
  Serial.begin(9600);
  while (!Serial);

  InitIMU();
  InitModel();

//INIT BLE 
  if (!BLE.begin()) {
    Serial.println("- Starting Bluetooth® Low Energy module failed!");
    while (1);
  }


//SET DEVICE AS PERIPHERAL
  BLE.setLocalName("Arduino Nano 33 BLE (Peripheral)");
//SET SERVICE AND ADVERTISE
  BLE.setAdvertisedService(gestureService);
  gestureService.addCharacteristic(gestureCharacteristic);
  BLE.addService(gestureService);
  gestureCharacteristic.writeValue(-1);
  BLE.advertise();

  Serial.println("Nano 33 BLE (Peripheral Device)");
  Serial.println(" ");


}

// THIS PERIPHERAL ========= RIGHTT RIGHT GESTURE 
// CENTRAL ============== LEFT LEFT GESTURE
void loop()
{
  BLEDevice central = BLE.central();
  Serial.println("- Discovering central device...");
  delay(500);

  if (central) {
    Serial.println("* Connected to central device!");
    Serial.print("* Device MAC address: ");
    Serial.println(central.address());
    Serial.println(" ");

    while (central.connected()) {
      if (gestureCharacteristic.written()) {
          gesture_left = gestureCharacteristic.value(); // READ FROM CENTRAL DEVICE LEFT
          gesture_right = gestureDetection(); // READ DATA FROM THIIIS ARDUINO 
          writeGesture(gesture_right, gesture_left);
       }
    }
    
    Serial.println("* Disconnected to central device!");
  }
}

// EDIT THIS GESTURE LIST TO ADD OR REMOVE GESTURES
void writeGesture(int gesture_right, int gesture_left){
  //GESTURE LIST
  switch(gesture_right){
    case GESTURE_STOP:
      if(gesture_left == GESTURE_STOP){
        Serial.println("This is GESTURE STOP");
      } else
      Serial.println("GESTURE WASNT RECOGNIZED");
      break;
      
    case GESTURE_AHEAD:
      if(gesture_left == GESTURE_AHEAD){
        Serial.println("This is GESTURE STOP");
      } else
      Serial.println("GESTURE WASNT AHEAD");
      break;

    case GESTURE_BACK:
      if(gesture_left == GESTURE_BACK){
        Serial.println("This is GESTURE BACK");
      } else
      Serial.println("GESTURE WASNT RECOGNIZED");
      break;

    default:
      Serial.println("GESTURE WASNT RECOGNIZED");
      break;

  }
}


void InitIMU(){
// initialize the IMU
  if (!IMU.begin())
  {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();
}


void InitModel(){
    // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}


int gestureDetection(){
float aX, aY, aZ, gX, gY, gZ;

int gesture = -1;

  // wait for significant motion
  while (samplesRead == numSamples)
  {
    if (IMU.accelerationAvailable())
    {
      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold)
      {
        // reset the sample read count
        samplesRead = 0;
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples)
  {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable())
    {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);
      
      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples)
      {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk)
        {
          Serial.println("Invoke failed!");
          while (1);
          return -2;
        }


        gesture = 0;
        // Loop through the output tensor values from the model
        for (int i = 0; i < NUM_GESTURES; i++)
        {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6);

          if ((tflOutputTensor->data.f[i] > tflOutputTensor->data.f[gesture]) && (tflOutputTensor->data.f[i] > modelThreshold))
          {
            gesture = i;
          }
        }
        Serial.println("CHOSEN RIGHT GESTURE : ");
        Serial.println(GESTURES[gesture]);
        return gesture;
      }

    }
  }
}
