from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit.visualization import timeline_drawer
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit import transpile
import copy

from gate_population import GeneratePopulation

class GenerateCompositeDD: 

    """

    A class to integrate a Dynamical Decoupling (DD) sequence into a provided QuantumCircuit.

    This class allows separate handling of idle times in different qubit wires, avoiding the 
    issue where pulse schedules default to the minimum pulse duration for all qubits. It 
    computes idle times between gates and inserts a DD sequence (which nets to the identity) 
    into each qubit wire individually. The result is a correctly stitched-together circuit, 
    even in the presence of multi-qubit gates, that allows for different-timed pulses in separate
    qubit wires. 

    Attributes:
    -----------
    circuit : QuantumCircuit
        the input quantum circuit to which DD will be applied.

    dd_sequence : list
        the DD sequence of gates (which should net to the identity) to apply.

    gate_duration : int
        duration for individual gates.

    verbose : bool
        if True, print additional information about gate timings.

    """


    def __init__(self, circuit, dd_sequence, gate_duration, verbose = True): 

        """

        initializes the class object

        Parameters:
        -----------
        circuit : QuantumCircuit
            the input quantum circuit.

        dd_sequence : list
            the DD sequence of gates to apply.

        gate_duration : int
            preset duration for individual gates before any DD sequence is added.
            NOTE* right now it is a constant -- this of course is not true and will
            be updated soon.

        verbose : bool, optional
            flag to print additional information (default is True).

        """

        self.circuit = circuit 
        self.dd_sequence = dd_sequence
        self.gate_duration = gate_duration
        self.verbose = verbose

    def generate_gate_timing(self, circuit): 

        """

        generates the timing information for all gates in the circuit.

        Parameters:
        -----------
        circuit : QuantumCircuit
            the quantum circuit to analyze.

        Returns:
        --------
        list
            a list of dictionaries containing gate names, qubit indices, and gate durations.

        """

        gate_timings = []

        for instruction in circuit.data:
            inst = instruction.operation
            qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]  
            
            duration = instruction.operation.duration if hasattr(inst, 'duration') else None

            gate_timings.append({
                'gate': inst.name,
                'qubits': qubits,
                'duration': duration,
            })

        if self.verbose: 
            print('{:<6} | {:<6} | {}'.format('Gate', 'Qubit', 'Duration'))
            for timing in gate_timings:
                print('{:<6} | {:<6} | {}'.format(timing['gate'],
                                                  str(timing['qubits']),
                                                  timing['duration']))
     
        return gate_timings

    def gate_timing_to_duration(self, circuit): 

        """

        converts the gate timing data into an array for future scheduling.

        Parameters:
        -----------
        circuit : QuantumCircuit
            the quantum circuit to get duration information for future scheduling

        Returns:
        --------
        list
            a list of tuples containing gate names, qubit indices, and gate durations.

        """

        verbose, self.verbose = self.verbose, False
        self.gate_timings = self.generate_gate_timing(circuit)
        self.verbose = verbose

        durations_list = []
        for gate in self.gate_timings: 
            durations_list.append((gate['gate'], gate['qubits'], self.gate_duration))

        return durations_list 

    def preliminary_circuit(self, show_circ = False): 

        """

        generates an scheduled circuit with invalid pulses of suuuper long duration. 
        this is purely to be able to extract the timing duration for the actual gates 
        before a valid DD sequence is added.

        Parameters:
        -----------
        show_circ : bool, optional
            if True, draw the circuit (default is True).

        Returns:
        --------
        QuantumCircuit
            the generated preliminary circuit.

        """

        self.durations_list = self.gate_timing_to_duration(self.circuit)
        durations_list = self.durations_list

        # fake DD sequence 
        fake_individual = [XGate(), XGate()]

        for unitary_gate in fake_individual: 
            pulse = (unitary_gate.name, None, 10000)
            if (pulse not in durations_list): 
                durations_list.append(pulse)

        durations = InstructionDurations(
            durations_list
        )

        pm = PassManager(
            [
                ALAPSchedule(durations), 
                DynamicalDecoupling(durations, fake_individual)
            ]
        )

        preliminary_circuit = pm.run(self.circuit)

        self.gate_timings = self.generate_gate_timing(preliminary_circuit)

        if show_circ: 
            def draw(circuit, filename = 'scheduled_circ.png'): 
                import os 

                scheduled = transpile( 
                    circuit, 
                    optimization_level = 0, 
                    instruction_durations = InstructionDurations(), 
                    scheduling_method = 'alap'
                )

                timeline_drawer(scheduled).savefig(f'img/{filename}')
                return os.system(f'open img/{filename}')
            draw(preliminary_circuit)

        return preliminary_circuit

    def calculate_qubit_intervals(self, qubit): 

        """

        calculates the idle time intervals for a specific qubit.

        Parameters:
        -----------
        qubit : int
            the qubit index for which to calculate idle intervals.

        Returns:
        --------
        list
            a list of dictionaries containing the gate pairs and the interval duration between them.

        """

        timings = [t for t in self.gate_timings if qubit in t['qubits']]
        intervals = []

        i = 0
        while i < len(timings) - 1: 
            curr_gate = timings[i]
            next_gate = timings[i+1]

            if next_gate['gate'] == 'delay' and qubit in next_gate['qubits']:
                delay_duration = next_gate['duration']

                if i + 2 < len(timings): 
                    next_gate_after_delay = timings[i+2]
                    intervals.append({ 
                        'between': (curr_gate['gate'], next_gate_after_delay['gate']), 
                        'interval': delay_duration
                    })
                    i += 2
                else: 
                    break 
            else: 
                i += 1

        return intervals 

    def idle_times_in_circuit(self): 

        """

        determines the idle times between gates for all qubits in the circuit.

        Returns:
        --------
        QuantumCircuit, list
            the preliminary circuit and a list of idle intervals for each qubit.

        """

        preliminary_circuit = self.preliminary_circuit(show_circ = False)

        idle_times = []
        for qubit_idx in range(preliminary_circuit.num_qubits): 
            qubit_intervals = self.calculate_qubit_intervals(qubit_idx)

            idle_times.append(qubit_intervals)

            if self.verbose: 
                string = f'Time intervals between gates for qubit {qubit_idx}:'
                print('-'*len(string))
                print(f"\n{string}")
                print(qubit_intervals)
                print('-'*len(string))

        return preliminary_circuit, idle_times 

    def compose_circuit(self, individual, delay_threshold, measure_all = True, show_circ = False): 

        """

        composes the final circuit with the DD sequence inserted based on idle times.

        Parameters:
        -----------
        individual : list
            the DD sequence to apply.

        delay_threshold : int
            threshold for adding a delay gate.

        measure_all : bool, optional
            if True, add measurements to all qubits (default is True).

        show_circ : bool, optional
            if True, draw the composed circuit (default is False).

        Returns:
        --------
        QuantumCircuit
            the composed quantum circuit.

        """

        def extract_wire_gates(circuit, wire_idx): 

            # extracts the gates in a qubit wire, including multi-qubit gates

            wire_ops = []
            for gate, qargs, cargs in circuit.data: 
                if any(qarg == circuit.qubits[wire_idx] for qarg in qargs): 
                    wire_ops.append((gate, qargs, cargs))
            return wire_ops 

        circuit, intervals = self.idle_times_in_circuit()
        duration = self.gate_timing_to_duration(self.circuit)

        circuits = []
        for i in range(circuit.num_qubits): 
            dd_durations = []
            for j in range(len(individual)): 
                padding = 0

                if len(intervals[i]) == 1: 
                    padding = intervals[i][0]['interval']
                if len(intervals[i]) > 1: 
                    padding = min(k['interval'] for k in intervals[i])

                pulse_duration = int(padding / len(individual) - 1)
                pulse = (individual[j].name, None, pulse_duration)

                if pulse not in dd_durations:
                    dd_durations.append(pulse)

            dd_qubit_durations = duration
            dd_qubit_durations.extend(dd_durations)
            durations = InstructionDurations(
                dd_qubit_durations
            )
            pm = PassManager(
                [
                    ALAPSchedule(durations), 
                    DynamicalDecoupling(durations, individual, qubits = [i])
                ]
            )
            circ_dd = pm.run(circuit)
            circuits.append(circ_dd)

        composed_qc = QuantumCircuit(QuantumRegister(circuit.num_qubits, 'q'))
        multi_qubit_gates = []

        for i in range(composed_qc.num_qubits): 
            for gate, qargs, cargs in extract_wire_gates(circuits[i], i): 
                gate_idxs = [composed_qc.find_bit(qarg)[0] for qarg in qargs]
                if len(gate_idxs) > 1 and gate_idxs not in multi_qubit_gates: 
                    multi_qubit_gates.append(gate_idxs)
                    composed_qc.append(gate, qargs, cargs)
                if len(gate_idxs) == 1: 
                    valid = True 
                    if gate.name == 'delay' and gate.duration >= delay_threshold: 
                        valid = False
                    if valid: 
                        composed_qc.append(gate, qargs, cargs)

        if measure_all: 
            composed_qc.measure_all()
        if show_circ: 
            import os 
            filename = 'composed_qc.png'
            composed_qc.draw('mpl', fold = -1).savefig(f'img/{filename}')

            os.system(f'open img/{filename}')

        return composed_qc


class GenerateConstantDD: 

    """

    This class is for generating DD circuits with a constant sequence duration 
    across all qubit wires and requires no stitching. This is basically what
    Qiskit's DynamicalDecoupling class does. Although built from first principles. 

    The `transpile_n_inverse` method at the end converts the scheduled circuit into 
    a form backends can understand, although additional backend-specific transpilations
    will still be required when actually running a job. 

    This method also appends the inverse of the original base circuit after the DD sequences. 
    This is to be able to assess the effectiveness of the DD circuits. The appended inverse of a circuit 
    -- in an ideal environment -- would return all the qubits to the |00...0> states. However, in the 
    presence of noise, some other states could appear -- signifying the quantum gates weren't perfect. A 
    DD sequence may improve the probabilty of yielding a |0...0> state, and this can be tested. 

    """  



    def __init__(self, circuit, dd_sequence, gate_duration, verbose = True): 

        self.circuit = circuit 
        self.dd_sequence = dd_sequence
        self.gate_duration = gate_duration
        self.verbose = verbose

    def generate_gate_timing(self, circuit): 

        gate_timings = []

        for instruction in circuit.data:
            inst = instruction.operation
            qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]  
            
            duration = instruction.operation.duration if hasattr(inst, 'duration') else None

            gate_timings.append({
                'gate': inst.name,
                'qubits': qubits,
                'duration': duration,
            })

        if self.verbose: 
            print('{:<12} | {:<12} | {}'.format('Gate', 'Qubit', 'Duration'))
            for timing in gate_timings:
                print('{:<12} | {:<12} | {}'.format(timing['gate'],
                                                  str(timing['qubits']),
                                                  timing['duration']))
     
        return gate_timings

    def gate_timing_to_duration(self, circuit): 

        verbose, self.verbose = self.verbose, False
        self.gate_timings = self.generate_gate_timing(circuit)
        self.verbose = verbose

        durations_list = []
        for gate in self.gate_timings: 
            durations_list.append((gate['gate'], gate['qubits'], self.gate_duration))

        return durations_list 

    def preliminary_circuit(self, show_circ = False): 

        self.durations_list = self.gate_timing_to_duration(self.circuit)
        durations_list = self.durations_list

        # fake DD sequence 
        fake_individual = [XGate(), XGate()]

        for unitary_gate in fake_individual: 
            pulse = (unitary_gate.name, None, 10000)
            if (pulse not in durations_list): 
                durations_list.append(pulse)

        durations = InstructionDurations(
            durations_list
        )

        pm = PassManager(
            [
                ALAPSchedule(durations), 
                DynamicalDecoupling(durations, fake_individual)
            ]
        )

        preliminary_circuit = pm.run(self.circuit)

        self.gate_timings = self.generate_gate_timing(preliminary_circuit)

        if show_circ: 
            def draw(circuit, filename = 'scheduled_circ.png'): 
                import os 

                scheduled = transpile( 
                    circuit, 
                    optimization_level = 0, 
                    instruction_durations = InstructionDurations(), 
                    scheduling_method = 'alap'
                )

                timeline_drawer(scheduled).savefig(f'img/{filename}')
                return os.system(f'open img/{filename}')
            draw(preliminary_circuit)

        return preliminary_circuit

    def calculate_qubit_intervals(self, qubit): 

        timings = [t for t in self.gate_timings if qubit in t['qubits']]
        intervals = []

        i = 0
        while i < len(timings) - 1: 
            curr_gate = timings[i]
            next_gate = timings[i+1]

            if next_gate['gate'] == 'delay' and qubit in next_gate['qubits']:
                delay_duration = next_gate['duration']

                if i + 2 < len(timings): 
                    next_gate_after_delay = timings[i+2]
                    intervals.append({ 
                        'between': (curr_gate['gate'], next_gate_after_delay['gate']), 
                        'interval': delay_duration
                    })
                    i += 2
                else: 
                    break 
            else: 
                i += 1

        return intervals 

    def idle_times_in_circuit(self, show_circ = False): 

        preliminary_circuit = self.preliminary_circuit(show_circ = False)

        if show_circ: 
            print("")
            print('-' * 66)
            print(preliminary_circuit)
            print('-' * 66)
            print("")

        string = 'Time intervals between gates for qubit 0:'
        if self.verbose:
            print('-'*(len(string) + 8))

        idle_times = []

        for qubit_idx in range(preliminary_circuit.num_qubits): 
            qubit_intervals = self.calculate_qubit_intervals(qubit_idx)

            idle_times.append(qubit_intervals)

            if self.verbose: 
                print(f"Time intervals between gates for qubit {qubit_idx}:")
                print(qubit_intervals)

        if self.verbose:
            print('-'*(len(string) + 8))

        return preliminary_circuit, idle_times 

    def minimum_pulse(self):

        circuit, idle_times = self.idle_times_in_circuit()
        minimum = []
        for interval_list in idle_times: 
            if interval_list:
                minimum.append([entry['interval'] for entry in interval_list])
        pulse_duration = int(min(minimum[0]) / len(self.dd_sequence))

        return pulse_duration 

    def scheduled_circuit(self, show_circ = False): 

        pulse_duration = self.minimum_pulse()
        base_durations = self.durations_list

        for unitary_gate in self.dd_sequence:
            pulse = (unitary_gate.name, None, pulse_duration)
            if pulse not in base_durations: 
                base_durations.append(pulse) 

        self.base_durations = base_durations

        durations = InstructionDurations(base_durations)

        pm = PassManager(
            [
                ALAPSchedule(durations), 
                DynamicalDecoupling(durations, self.dd_sequence)
            ]
        )

        composed_qc = pm.run(self.circuit)

        if show_circ: 
            def draw(circuit, filename = 'scheduled_circ_constant.png'): 
                import os 

                scheduled = transpile( 
                    circuit, 
                    optimization_level = 0, 
                    instruction_durations = InstructionDurations(), 
                    scheduling_method = 'alap'
                )

                timeline_drawer(scheduled).savefig(f'img/{filename}')
                return os.system(f'open img/{filename}')
            draw(composed_qc)

        return composed_qc

    def transpile_n_inverse(self, save_circ = True, open_circ = False): 

        circuit = self.scheduled_circuit()

        circuit.data = [ 
            instruction for instruction in circuit.data 
            if instruction[0].name not in {'barrier', 'id'}
        ]

        final_circuit = QuantumCircuit(circuit.num_qubits)

        for gate_name, qargs, cargs in circuit.data:
            qubits = [circuit.find_bit(qarg)[0] for qarg in qargs]
            if (gate_name.name != 'barrier' and 
                gate_name.name != 'measure' and 
                gate_name.name != 'i'):
     
                if len(qubits) >= 2 and len(qubits) < 3: 
                    getattr(final_circuit, gate_name.name)(*qubits)
                elif len(qubits) < 2: 
                    getattr(final_circuit, gate_name.name)(qubits[0])

        base_circ = copy.deepcopy(self.circuit)
        inverse_circ = base_circ.inverse()
        final_circuit.compose(inverse_circ, inplace=True)
        final_circuit.measure_all()
        self.base_durations.extend([('measure', None, 1000)])

        if save_circ: 
            import os 

            filename = 'transpile_n_inverse_circ.png'
            final_circuit.draw('mpl', fold = -1).savefig(f'img/{filename}')
            if open_circ: 
                os.system(f'open img/{filename}')

        return final_circuit, self.base_durations

        










if __name__ == '__main__': 

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)

    circuit.barrier()
    print(circuit.draw())

    class_population = GeneratePopulation(basis_gates=[XGate(), YGate(), ZGate()], 
                                        num_individuals = 10, 
                                        length_range = [4, 6],
                                        max_gate_reps = 6, 
                                        basis_names = ['x', 'y', 'z'],
                                        verbose = False
    )

    population = class_population.population()

    #class_ = GenerateCompositeDD(circuit = circuit, dd_sequence = population[0], 
    #                           gate_duration = 50, verbose = False)
    #class_.compose_circuit(individual = population[0], delay_threshold = 15)

    class_ = GenerateConstantDD(circuit = circuit, dd_sequence = population[0], 
                                gate_duration = 50, verbose = True)
    final_circuit = class_.transpile_n_inverse()







        

