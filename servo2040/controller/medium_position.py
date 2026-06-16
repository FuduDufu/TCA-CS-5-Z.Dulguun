import serial
import time

PORT = "COM4"      # <-- өөрийн COM port-оо тавь
BAUD = 115200

CMD_SET = 0x53 | 0x80  # 0xD3

def encode_set(start, values):
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

def send_set(ser, start, values, label=""):
    pkt = encode_set(start, values)
    if label:
        print(f"{label}: {to_hex(pkt)}")
    else:
        print(to_hex(pkt))
    ser.write(pkt)

# P00..P17 хүртэлх бүх servo-ийн midpoint values
CENTER_ALL = [
    1450,  # P00 R31
    1500,  # P01 R32
    1500,  # P02 R33
    1540,  # P03 L31
    1480,  # P04 L32
    1525,  # P05 L33
    1525,  # P06 R21
    1480,  # P07 R22
    1550,  # P08 R23
    1525,  # P09 L21
    1530,  # P10 L22
    1430,  # P11 L23
    1500,  # P12 R11
    1760,  # P13 R12
    1470,  # P14 R13
    1525,  # P15 L11
    1500,  # P16 L12
    1470,  # P17 L13
]

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # board reset-д бага хугацаа өгнө

    try:
        # 1) Relay ON
        send_set(ser, 26, [1], "RELAY ON")
        time.sleep(1.0)

        # 2) Бүх servo-г нэг packet-аар голын байрлал руу
        send_set(ser, 0, CENTER_ALL, "CENTER ALL P00-P17")
        time.sleep(2.0)

        print("Done. Robot all legs moved to center position.")

        # Хэрэв relay-г унтраахыг хүсвэл доорх 2 мөрийг uncomment хийнэ
        send_set(ser, 26, [0], "RELAY OFF")
        time.sleep(0.5)

    finally:
        ser.close()

if __name__ == "__main__":
    main()