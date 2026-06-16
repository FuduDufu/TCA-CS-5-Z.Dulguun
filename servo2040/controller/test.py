import serial
import time

# COM port-оо энд тавина
PORT = "COM4"
BAUD = 115200

def encode_set(start, values):
    CMD_SET = 0x53 | 0x80

    data = bytearray()
    data.append(CMD_SET)
    data.append(start & 0x7F)
    data.append(len(values) & 0x7F)

    for v in values:
        data.append(v & 0x7F)
        data.append((v >> 7) & 0x7F)

    return data


def to_hex(data):
    return " ".join(f"{b:02X}" for b in data)


ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)  # board reset-д хугацаа өгнө

# 👉 P15 дээр servo 1500us
packet = encode_set(15, [1500])

print("SEND:", to_hex(packet))
ser.write(packet)

time.sleep(0.5)

ser.close()