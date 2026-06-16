import serial
import time

PORT = "COM4q"      # <-- өөрийн COM port-оо тавь
BAUD = 115200

CMD_GET = 0x47 | 0x80  # 0xC7

def encode_get(start, count):
    return bytearray([
        CMD_GET,
        start & 0x7F,
        count & 0x7F
    ])

def decode_values(data):
    if len(data) < 3:
        return None, None, []

    cmd = data[0]
    start = data[1]
    count = data[2]

    values = []
    expected_len = 3 + count * 2

    if len(data) < expected_len:
        return cmd, start, values

    idx = 3
    for _ in range(count):
        low = data[idx] & 0x7F
        high = data[idx + 1] & 0x7F
        value = low | (high << 7)
        values.append(value)
        idx += 2

    return cmd, start, values

def to_hex(data):
    return " ".join(f"{b:02X}" for b in data)

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

try:
    while True:
        # P18..P23 => 6 touch sensor
        packet = encode_get(18, 6)
        ser.reset_input_buffer()
        ser.write(packet)

        # reply: [G][18][6][v1][v2][v3][v4][v5][v6]
        resp = ser.read(3 + 6 * 2)

        cmd, start, values = decode_values(resp)

        if len(values) == 6:
            print(
                f"TS_R3={values[0]}  "
                f"TS_L3={values[1]}  "
                f"TS_R2={values[2]}  "
                f"TS_L2={values[3]}  "
                f"TS_R1={values[4]}  "
                f"TS_L1={values[5]}"
            )
        else:
            print("RECV ERROR:", to_hex(resp))

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    ser.close()