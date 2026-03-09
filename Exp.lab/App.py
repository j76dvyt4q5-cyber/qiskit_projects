from flask import Flask, jsonify, request, send_from_directory
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
import random
import os

app = Flask(__name__)

# Allow local frontend to call the API
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ── QUANTUM FUNCTIONS ──────────────────────────────────────────

def measure_qc_card_cout(qc, card, c_out):
    qc.measure(card, c_out)
    sim = AerSimulator()
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc, shots=1).result()
    return result.get_counts()

def apply_table_gate(qc, control_table, table, card, target_hand, theta, table_card_gate):
    if table_card_gate == "CRY":
        qc.cry(theta, table[control_table], card[target_hand])
    elif table_card_gate == "CX":
        qc.cx(table[control_table], card[target_hand])
    elif table_card_gate == "CZ":
        qc.cz(table[control_table], card[target_hand])

def apply_self_gate(qc, card, self_gate, target_index, strength=None):
    if self_gate == "H":
        qc.h(card[target_index])
    elif self_gate == "Z":
        qc.z(card[target_index])
    elif self_gate == "X":
        qc.x(card[target_index])
    elif self_gate == "RY":
        if strength is None:
            raise ValueError("Strength required for RY gate")
        qc.ry(strength, card[target_index])

def quantum_index(card, table, c_out, table_mods=None, self_mods=None):
    qc = QuantumCircuit(card, table, c_out)
    qc.h(card)
    qc.h(table)
    if table_mods:
        for ct, th, tcg, theta in table_mods:
            apply_table_gate(qc, ct, table, card, th, theta, tcg)
    if self_mods:
        for sg, ti, theta in self_mods:
            apply_self_gate(qc, card, sg, ti, theta)
    counts = measure_qc_card_cout(qc, card, c_out)
    bitstring = list(counts.keys())[0]
    return int(bitstring, 2)

def fresh_deck():
    faces = ['Ace','2','3','4','5','6','7','8','9','10','Jack','Queen','King']
    deck = faces * 4
    random.shuffle(deck)
    return deck

def draw_card(deck, card, table, c_out, table_mods=None, self_mods=None):
    if not deck:
        deck.extend(fresh_deck())
    while True:
        idx = quantum_index(card, table, c_out, table_mods, self_mods)
        if idx < len(deck):
            drawn = deck[idx]
            deck.pop(idx)
            return drawn
        table_mods = None
        self_mods = None

def card_to_value(card):
    if card in ['Jack', 'Queen', 'King']: return 10
    elif card == 'Ace': return 11
    elif card.isdigit(): return int(card)
    return 0

def add_card_to_total(card, total, ace_count):
    if card == 'Ace':
        total += 11
        ace_count += 1
    else:
        total += card_to_value(card)
    while total > 21 and ace_count > 0:
        total -= 10
        ace_count -= 1
    return total, ace_count

def is_blackjack(total, hand):
    return total == 21 and len(hand) == 2

# ── GAME STATE ─────────────────────────────────────────────────

state = {}

def init_game():
    global state
    card_reg  = QuantumRegister(4, "card")
    table_reg = QuantumRegister(2, "table")
    c_out_reg = ClassicalRegister(4, "c_out")
    deck = fresh_deck()

    c0 = draw_card(deck, card_reg, table_reg, c_out_reg)
    pt, ac = add_card_to_total(c0, 0, 0)
    c1 = draw_card(deck, card_reg, table_reg, c_out_reg)
    pt, ac = add_card_to_total(c1, pt, ac)

    hand = [c0, c1]
    state = {
        'card_reg':        card_reg,
        'table_reg':       table_reg,
        'c_out_reg':       c_out_reg,
        'deck':            deck,
        'player_hand':     hand,
        'player_total':    pt,
        'ace_count':       ac,
        'table_modifiers': [],
        'self_modifiers':  [],
        'ultimate_used':   False,
        'dealer_hand':     [],
        'dealer_total':    0,
        'phase':           'playing',
        'result':          None,
        'messages':        []
    }

    if is_blackjack(pt, hand):
        state['phase']  = 'done'
        state['result'] = 'player_blackjack'
        state['messages'] = [f'You were dealt {c0} and {c1} — Blackjack! You win instantly!']
    else:
        state['messages'] = [f'Dealt {c0} and {c1}. Your total: {pt}.']

def state_json():
    return jsonify({
        'player_hand':     state['player_hand'],
        'player_total':    state['player_total'],
        'dealer_hand':     state['dealer_hand'],
        'dealer_total':    state['dealer_total'],
        'phase':           state['phase'],
        'result':          state['result'],
        'ultimate_used':   state['ultimate_used'],
        'deck_remaining':  len(state['deck']),
        'self_modifiers':  [(m[0], m[1]) for m in state['self_modifiers']],
        'table_modifiers': [(m[0], m[1], m[2]) for m in state['table_modifiers']],
        'messages':        state['messages']
    })

# ── ROUTES ─────────────────────────────────────────────────────

@app.route('/')
def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quantum_blackjack.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/new_game', methods=['POST', 'OPTIONS'])
def new_game():
    if request.method == 'OPTIONS':
        return '', 204
    init_game()
    return state_json()

@app.route('/api/action', methods=['POST', 'OPTIONS'])
def action():
    if request.method == 'OPTIONS':
        return '', 204
    if not state:
        return jsonify({'error': 'No game in progress'}), 400
    if state['phase'] == 'done':
        return jsonify({'error': 'Game over — start a new game'}), 400

    data = request.get_json()
    act  = data.get('action')
    state['messages'] = []

    if act == 'hit':
        card_drawn = draw_card(
            state['deck'], state['card_reg'], state['table_reg'], state['c_out_reg'],
            state['table_modifiers'] or None,
            state['self_modifiers'] or None
        )
        state['player_hand'].append(card_drawn)
        state['player_total'], state['ace_count'] = add_card_to_total(
            card_drawn, state['player_total'], state['ace_count']
        )
        state['table_modifiers'].clear()
        state['self_modifiers'].clear()

        if state['player_total'] > 21:
            state['phase']  = 'done'
            state['result'] = 'bust'
            state['messages'].append(f'You drew {card_drawn}. Bust! Total: {state["player_total"]}.')
        else:
            state['messages'].append(
                f'You drew {card_drawn}. Total: {state["player_total"]}. '
                f'{len(state["deck"])} cards remaining.'
            )

    elif act == 'stand':
        state['messages'].append(f'You stand with {state["player_total"]}.')
        _run_dealer()

    elif act == 'self_gate':
        gate  = data.get('gate')
        qubit = int(data.get('qubit'))
        theta = data.get('theta')
        state['self_modifiers'].append((gate, qubit, theta))
        label = f' (strength {theta})' if theta is not None else ''
        state['messages'].append(f'{gate} gate queued on card qubit {qubit}{label}. Fires on next hit.')

    elif act == 'table_gate':
        control = int(data.get('control'))
        target  = int(data.get('target'))
        gate    = data.get('gate')
        theta   = data.get('theta')
        state['table_modifiers'].append((control, target, gate, theta))
        state['messages'].append(f'{gate} queued from table[{control}] → card[{target}]. Fires on next hit.')

    elif act == 'ultimate':
        if state['ultimate_used']:
            state['messages'].append('Ultimate already used this round.')
        else:
            state['table_modifiers'].append((0, 0, 'CX', None))
            state['table_modifiers'].append((1, 1, 'CX', None))
            state['ultimate_used'] = True
            state['messages'].append('Ultimate activated! CX entanglement on both pairs fires on next hit.')

    return state_json()

def _run_dealer():
    dealer_ace = 0
    state['phase'] = 'dealer'

    while state['dealer_total'] < 17:
        drawn = draw_card(state['deck'], state['card_reg'], state['table_reg'], state['c_out_reg'])
        state['dealer_hand'].append(drawn)
        state['dealer_total'], dealer_ace = add_card_to_total(
            drawn, state['dealer_total'], dealer_ace
        )

    state['phase'] = 'done'
    pt = state['player_total']
    dt = state['dealer_total']

    state['messages'].append(f'Dealer: {", ".join(state["dealer_hand"])} — total {dt}.')

    if is_blackjack(dt, state['dealer_hand']):
        if is_blackjack(pt, state['player_hand']):
            state['result'] = 'tie'
            state['messages'].append('Both have Blackjack — tie!')
        else:
            state['result'] = 'dealer_wins'
            state['messages'].append('Dealer has Blackjack.')
    elif dt > 21:
        state['result'] = 'player_wins'
        state['messages'].append('Dealer busts! You win!')
    elif dt > pt:
        state['result'] = 'dealer_wins'
        state['messages'].append(f'Dealer wins with {dt} vs your {pt}.')
    elif dt < pt:
        state['result'] = 'player_wins'
        state['messages'].append(f'You win with {pt} vs dealer\'s {dt}!')
    else:
        state['result'] = 'tie'
        state['messages'].append(f'Tie — both at {pt}.')

# ── ENTRY ──────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n  Quantum Blackjack server starting...')
    print('  Open http://localhost:5000 in your browser.\n')
    app.run(debug=False, port=5000)