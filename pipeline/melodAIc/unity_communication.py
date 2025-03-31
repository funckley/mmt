from pythonosc import dispatcher, osc_server, udp_client
from rule_based_system import RuleBasedSystem
from short_term_memory import ShortTermMemory

class UnityCommunication:
    def __init__(self, ip="127.0.0.1", port=8000, client_ip="127.0.0.1", client_port=9000):
        """Initialize the Unity communication and OSC server."""
        self.rule_system = RuleBasedSystem()
        self.short_term_memory = ShortTermMemory()
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/change_tempo", self.change_tempo)
        self.dispatcher.map("/change_dynamics", self.change_dynamics)
        self.dispatcher.map("/change_expression", self.change_expression)
        self.dispatcher.map("/pitch_shift", self.pitch_shift)
        
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
        self.client = udp_client.SimpleUDPClient(client_ip, client_port)

    def start_server(self):
        """Start the OSC server to receive commands."""
        print(f"Serving on {self.server.server_address}")
        self.server.serve_forever()

    def change_tempo(self, unused_addr, tempo):
        """Handle change_tempo command."""
        response = self.rule_system.change_tempo({"tempo": tempo})
        self.client.send_message("/response", response["message"])

    def change_dynamics(self, unused_addr, dynamics):
        """Handle change_dynamics command."""
        response = self.rule_system.change_dynamics({"dynamics": dynamics})
        self.client.send_message("/response", response["message"])

    def change_expression(self, unused_addr, expression):
        """Handle change_expression command."""
        response = self.rule_system.change_expression({"expression": expression})
        self.client.send_message("/response", response["message"])

    def pitch_shift(self, unused_addr, shift_amount):
        """Handle pitch_shift command."""
        response = self.rule_system.pitch_shift({"shift_amount": shift_amount})
        self.client.send_message("/response", response["message"])

if __name__ == "__main__":
    unity_comm = UnityCommunication()
    unity_comm.start_server()