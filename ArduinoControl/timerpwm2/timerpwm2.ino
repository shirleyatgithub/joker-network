#include <FlexiTimer2.h>
String readString;

int pwmpin=10;


int used=50;
int interflg=0;

int hz = 100;

int lastsgn=0;
int lastsgnvalue=0;

void event2()
{
    analogWrite(pwmpin, 0);

//    Serial.println(lastsgnvalue);

    interflg = 1;
    lastsgnvalue=64;

    FlexiTimer2::stop();
    
}
 
 
void setup() {

  Serial.begin(9600);

    pinMode(pwmpin,OUTPUT);

    FlexiTimer2::set(50,1.0/1000,event2);
//    FlexiTimer2::start();
}
void loop() {

  while (!Serial.available()) {

    analogWrite(pwmpin, lastsgnvalue);

    
    } // wait for data to arrive
  // serial read section
  while (Serial.available()) // this will be skipped if no data present, leading to
                             // the code sitting in the delay function below
  {
    delayMicroseconds(100);  //delay to allow buffer to fill 
    if (Serial.available() >0)
    {
      char c = Serial.read();  //gets one byte from serial buffer
      readString = c; //makes the string readString

    }
  }
  if (readString.length() >0)
  {
    int num = readString.toInt();
    lastsgn=num;


      if (num == 1){
       
        if (interflg==0)
        {

          
          analogWrite(pwmpin, 255);   
          lastsgnvalue=255;

          FlexiTimer2::start();

        }
     
      }else{
        analogWrite(pwmpin, 0);
        lastsgnvalue=0;
        interflg=0;
        }
  }
  
}
