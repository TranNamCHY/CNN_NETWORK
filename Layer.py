class Layer:
    def __init__(self, name):
        self.name = name
    def forward(self, input):
        print("Forward func not been override at " + self.name + " layer !.")
    def int8_forward(self, input):
        print("8bit Forward func not been override at " + self.name + " layer !.")
    def save_weight(self):
        return
    def load_weight_by_name(self):
        return