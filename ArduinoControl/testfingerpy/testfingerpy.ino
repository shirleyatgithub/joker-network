int motorPin = 13;
String readString;

void setup()
{

  Serial.begin(9600);  // initialize serial communications at 9600 bps

}

void loop()
{
  while (!Serial.available()) {} // wait for data to arrive
  // serial read section
  while (Serial.available()) // this will be skipped if no data present, leading to
                             // the code sitting in the delay function below
  {
    delay(30);  //delay to allow buffer to fill 
    if (Serial.available() >0)
    {
      char c = Serial.read();  //gets one byte from serial buffer
      readString = c; //makes the string readString
    }
  }
  if (readString.length() >0)
  {
//    Serial.println(readString.length());
//    char flag = readString.substring(1, readString.length()); ;
    Serial.print("Arduino received: ");  
    Serial.println(readString); //see what was received
    int num = readString.toInt();
    Serial.print("this is num "); 
    Serial.println(num); 
    if (num == 1){
//      Serial.println("this is 1 "); 
      digitalWrite(motorPin, LOW);
//   Serial.println("hello");
//   delay(1000);
   
      }
      else 
      {
//        Serial.println("this is 0"); 
        digitalWrite(motorPin, HIGH);
//   Serial.println("world");
//   delay(1000);
        }

    
//    Serial.print(" flag: ");  
//    Serial.println(flag);
  }

  delay(500);

  // serial write section

  char ard_sends = '1';
  Serial.print("Arduino sends: ");
  Serial.println(ard_sends);
  Serial.print("\n");
  Serial.flush();
//  Serial.flushInput();

//  while (Serial.available()>=0){}
}
