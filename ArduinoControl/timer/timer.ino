#include <FlexiTimer2.h>
 
int ledpin=2;
int buttonpin=10;
void event()
{
    static boolean output=HIGH;
    digitalWrite(ledpin,output);
    output=!output;
}
 
void setup() {
    pinMode(ledpin,OUTPUT);
    pinMode(buttonpin,OUTPUT);
    FlexiTimer2::set(50,1.0/1000,event);
    FlexiTimer2::start();
}
void loop() {
//    for(int i=0;i<255;i++){
//      analogWrite(ledpin,i);
//      delay(5);
//    }
//    for(int i=255;i>0;i--){
//      analogWrite(ledpin,i);
//      delay(5);
//    }
}
