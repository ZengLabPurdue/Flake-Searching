# Usart Library
import serial
import time as t

ERROR_CODES = {
    "11214": "Motor timeout (turret failed to reach position)",
    "10101": "Command format error",
    "10102": "Parameter error",
    "10103": "Command not executable",
}

NUM_OBJECTIVES = 5

class TurretController:
    """
    API class for controlling BX-REMCB turret controller
    """

    def __init__(self, port: str):
        """
        Initialize the serial port and log in to the controller
        """
        print("Starting BX-REMCB controller")
        
        # Initialize serial connection
        self.Usart = serial.Serial(
            port=port,
            baudrate=19200,                     # BX-REMCB default baudrate
            bytesize=serial.EIGHTBITS,             # 8 data bits
            parity=serial.PARITY_EVEN,             # Even parity
            stopbits=serial.STOPBITS_TWO,          # 2 stop bits
            timeout=1                             # 1 s read timeout
        )
        
        # Check CTS status
        if self.Usart.getCTS():
            print("CTS asserted (ready to receive)")
        else:
            print("CTS de-asserted (not ready)")
        
        t.sleep(1)
        
        # Log in to the controller
        self.Usart.write('1LOG IN\r\n'.encode())
        t.sleep(0.5)
        current_response = self.Usart.readline()
        
        if current_response == b'1LOG +\r\n':
            print("LOG IN successful!")
        else:
            print(f"Login failed. Response: {current_response}")

    def send_command(self, command, timeout=5):

        self.Usart.reset_input_buffer()

        full_cmd = f"{command}\r\n".encode()

        print(f"Sending: {command}")
        self.Usart.write(full_cmd)

        start = t.time()

        while True:

            if t.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for response to {command}")

            response = self.Usart.readline()

            if response:

                response = response.decode().strip()
                print(f"Received: {response}")

                # ERROR
                if "!" in response:
                    raise Exception(f"Controller error: {response}")

                # SUCCESS ACK
                if "+" in response:
                    return response

                # QUERY RESPONSE (value returned)
                if "?" in command:
                    return response
            
    def check_if_log_in(self):
        self.Usart.write("1LOG?\r\n".encode())
        t.sleep(0.5)
        response = self.Usart.readline()
        print(response)
        
        # Check for valid response indicating logged in status
        if response and b'1LOG IN' in response:
            print("Controller is logged in")
            return True
        else:
            print("Controller is not logged in")
            return False
    
    def turn_to_position(self, value):

        if not 1 <= value <= NUM_OBJECTIVES:
            raise ValueError("Turret position must be 1–6")

        print(f"Moving turret to position {value}")

        response = self.send_command(f"1OB {value}")

        print("Move complete:", response)
    
    def check_position(self):

        response = self.send_command("1OB?")

        try:
            parts = response.split()
            position = int(parts[1])
            print("Current position:", position)
            return position
        except:
            print("Unexpected response:", response)
            return None
    
    def close(self):
        try:
            # Log out
            self.Usart.write('1LOG OUT\r\n'.encode())

            t.sleep(0.5)

            logout_response = self.Usart.readline()
            print(f"Logout response: {logout_response}")
            
            # Close serial port
            self.Usart.close()
            print("Serial port closed")
        except Exception as e:
            print(f"Error during close: {e}")

def test_run():
    try:
        controller = TurretController(port='COM4')
        
        # Test check_if_log_in
        print("\n--- Testing check_if_log_in ---")
        controller.check_if_log_in()
        
        # Test check_position
        print("\n--- Testing check_position ---")
        controller.check_position()

        t.sleep(3)
        # move the six-place nosepiece to position 2
        controller.turn_to_position(1)

        t.sleep(3)
        # move the six-place nosepiece to position 2
        controller.turn_to_position(2)

        t.sleep(3)
        # move the six-place nosepiece to position 2
        controller.turn_to_position(3)

        print("Write done")
        
        controller.close()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
        if 'controller' in locals():
            controller.close()


if __name__ == "__main__":
    test_run()