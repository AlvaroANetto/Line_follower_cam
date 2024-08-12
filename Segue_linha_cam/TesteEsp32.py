from machine import Pin
import time

# Configura o pino do LED integrado (normalmente GPIO2 na maioria das placas ESP32)
led = Pin(2, Pin.OUT)

# Liga o LED
led.value(1)

# Mantém o LED ligado por 5 segundos
time.sleep(5)

# Desliga o LED
led.value(0)
