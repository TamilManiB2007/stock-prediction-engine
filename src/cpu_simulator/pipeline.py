import random
import time

class CPUInstruction:
    def __init__(self, opcode, operands="", pc=0):
        self.opcode = opcode
        self.operands = operands
        self.pc = pc
        self.result = None

class PipelinedCPU:
    def __init__(self, branch_predictor=None):
        self.pc = 0
        self.registers = [0] * 32
        self.memory = [0] * 1024
        self.branch_predictor = branch_predictor
        
        # Pipeline stages
        self.if_stage = None
        self.id_stage = None
        self.ex_stage = None
        self.mem_stage = None
        self.wb_stage = None
        
        # Statistics
        self.cycles = 0
        self.instructions_executed = 0
        self.branch_mispredictions = 0
        self.pipeline_stalls = 0
        
    def generate_trading_workload(self, strategy="moving_average"):
        """Generate CPU instructions based on trading strategy"""
        instructions = []
        
        if strategy == "moving_average":
            # Predictable branch pattern
            for i in range(50):
                instructions.extend([
                    CPUInstruction("LOAD", f"R1, price[{i}]", i*4),
                    CPUInstruction("ADD", "R2, R1, R2", i*4+1),
                    CPUInstruction("CMP", "R2, threshold", i*4+2),
                    CPUInstruction("BEQ", "R2, 0, SELL", i*4+3),  # Predictable branch
                ])
        
        elif strategy == "momentum":
            # Unpredictable branch pattern
            for i in range(50):
                instructions.extend([
                    CPUInstruction("LOAD", f"R1, momentum[{i}]", i*4),
                    CPUInstruction("CMP", "R1, 0", i*4+1),
                    CPUInstruction("BGT", "R1, BUY", i*4+2),  # Data-dependent branch
                    CPUInstruction("BLT", "R1, SELL", i*4+3),  # Unpredictable
                ])
        
        return instructions
    
    def simulate_pipeline(self, instructions):
        """Simulate pipeline execution"""
        instruction_queue = instructions.copy()
        
        while instruction_queue or any([self.if_stage, self.id_stage, self.ex_stage, self.mem_stage, self.wb_stage]):
            self.cycles += 1
            
            # Write Back
            if self.wb_stage:
                self.instructions_executed += 1
                self.wb_stage = None
            
            # Memory Access
            self.wb_stage = self.mem_stage
            self.mem_stage = None
            
            # Execute
            if self.ex_stage:
                if "B" in self.ex_stage.opcode:  # Branch instruction
                    # Simulate branch prediction
                    predicted = self.branch_predictor.predict() if self.branch_predictor else True
                    actual = random.choice([True, False])  # Simulate actual branch outcome
                    
                    if predicted != actual:
                        self.branch_mispredictions += 1
                        self.pipeline_stalls += 2  # Pipeline flush penalty
            
            self.mem_stage = self.ex_stage
            self.ex_stage = None
            
            # Decode
            self.ex_stage = self.id_stage
            self.id_stage = None
            
            # Fetch
            if instruction_queue:
                self.id_stage = self.if_stage
                self.if_stage = instruction_queue.pop(0)
        
        return self.get_performance_metrics()
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        ipc = self.instructions_executed / self.cycles if self.cycles > 0 else 0
        branch_miss_rate = (self.branch_mispredictions / self.instructions_executed * 100) if self.instructions_executed > 0 else 0
        
        return {
            "cycles": self.cycles,
            "instructions": self.instructions_executed,
            "ipc": round(ipc, 2),
            "branch_mispredictions": self.branch_mispredictions,
            "branch_miss_rate": round(branch_miss_rate, 2),
            "pipeline_stalls": self.pipeline_stalls
        }

class BranchPredictor:
    def __init__(self, predictor_type="static"):
        self.predictor_type = predictor_type
        self.prediction_table = {}
        self.global_history = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def predict(self, pc=0):
        """Predict branch outcome"""
        self.total_predictions += 1
        
        if self.predictor_type == "static":
            return True  # Always taken
        elif self.predictor_type == "1bit":
            return self.prediction_table.get(pc, True)
        elif self.predictor_type == "2bit":
            counter = self.prediction_table.get(pc, 2)
            return counter >= 2
        else:
            return random.choice([True, False])
    
    def update(self, pc, actual):
        """Update predictor state"""
        if actual == self.predict(pc):
            self.correct_predictions += 1
        
        if self.predictor_type == "1bit":
            self.prediction_table[pc] = actual
        elif self.predictor_type == "2bit":
            counter = self.prediction_table.get(pc, 2)
            if actual and counter < 3:
                self.prediction_table[pc] = counter + 1
            elif not actual and counter > 0:
                self.prediction_table[pc] = counter - 1
    
    def get_accuracy(self):
        """Get prediction accuracy"""
        return (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
