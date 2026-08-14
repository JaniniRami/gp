// ProactMAD -- DRV8871 linear-actuator firmware (ESP32 Arduino core 3.x)
// IN1 (open / advance)  -> GPIO 27
// IN2 (close / retract) -> GPIO 26
//
// Serial @ 115200. One command per line.
//   o / open / extend     open until STOP (manual)
//   c / close / retract   close until STOP (manual)
//   s / stop              coast/stop
//   ADVANCE               open for ADVANCE_MS then stop (controller fire)
//   RETRACT               close for RETRACT_MS then stop (controller retract)
//   0-255                 set PWM speed
//   ?                     help

const int IN1 = 27;
const int IN2 = 26;
const int pwmFreq = 5000;
const int pwmResolution = 8;

const unsigned long ADVANCE_MS = 10000;
const unsigned long RETRACT_MS = 10000;

int motorSpeed = 180;

enum Move : uint8_t { IDLE = 0, OPENING = 1, CLOSING = 2 };
Move move = IDLE;
unsigned long moveUntil = 0;

void stopMotor() {
  ledcWrite(IN1, 0);
  ledcWrite(IN2, 0);
}

void openMotor(int speed) {
  speed = constrain(speed, 0, 255);
  ledcWrite(IN2, 0);
  ledcWrite(IN1, speed);
}

void closeMotor(int speed) {
  speed = constrain(speed, 0, 255);
  ledcWrite(IN1, 0);
  ledcWrite(IN2, speed);
}

void printHelp() {
  Serial.println();
  Serial.println("ProactMAD DRV8871");
  Serial.println("  o/open     OPEN  (manual, until s)");
  Serial.println("  c/close    CLOSE (manual, until s)");
  Serial.println("  s/stop     STOP");
  Serial.println("  ADVANCE    open 10s then stop");
  Serial.println("  RETRACT    close 10s then stop");
  Serial.print("  speed=");
  Serial.println(motorSpeed);
}

void setup() {
  Serial.begin(115200);
  delay(400);
  ledcAttach(IN1, pwmFreq, pwmResolution);
  ledcAttach(IN2, pwmFreq, pwmResolution);
  stopMotor();
  Serial.println("READY ProactMAD");
  printHelp();
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.replace("\r", "");
  String low = cmd;
  low.toLowerCase();
  if (low.length() == 0) return;

  if (low == "o" || low == "e" || low == "open" || low == "extend") {
    move = IDLE;
    openMotor(motorSpeed);
    Serial.println("OK OPEN");
    return;
  }
  if (low == "c" || low == "r" || low == "close") {
    move = IDLE;
    closeMotor(motorSpeed);
    Serial.println("OK CLOSE");
    return;
  }
  if (low == "s" || low == "stop") {
    move = IDLE;
    stopMotor();
    Serial.println("OK STOP");
    return;
  }
  if (low == "advance") {
    openMotor(motorSpeed);
    move = OPENING;
    moveUntil = millis() + ADVANCE_MS;
    Serial.println("OK ADVANCE");
    return;
  }
  if (low == "retract") {
    closeMotor(motorSpeed);
    move = CLOSING;
    moveUntil = millis() + RETRACT_MS;
    Serial.println("OK RETRACT");
    return;
  }
  if (low == "?" || low == "help" || low == "h") {
    printHelp();
    return;
  }

  bool digits = true;
  for (unsigned i = 0; i < low.length(); i++) {
    if (!isDigit(low[i])) {
      digits = false;
      break;
    }
  }
  if (digits) {
    int v = low.toInt();
    if (v < 0 || v > 255) {
      Serial.println("ERR speed 0-255");
      return;
    }
    motorSpeed = v;
    Serial.print("OK SPEED ");
    Serial.println(motorSpeed);
    return;
  }

  Serial.print("ERR unknown ");
  Serial.println(cmd);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }
  if (move != IDLE && millis() >= moveUntil) {
    stopMotor();
    move = IDLE;
    Serial.println("OK HOLD");
  }
}
