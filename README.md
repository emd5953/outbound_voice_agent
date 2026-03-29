# Pizza Order Voice Agent 🍕📞

An autonomous voice agent that places outbound phone calls to order pizza delivery. It navigates a legacy IVR menu, waits on hold, then converses with a human employee to place a complete order — all driven by a JSON payload.

## How It Works

1. Receives an order payload via `POST /place-order`
2. Initiates an outbound call through Twilio
3. Navigates the restaurant's IVR (automated phone menu) using DTMF tones and speech
4. Detects when a human employee picks up after hold
5. Converses naturally to place the order, handling substitutions, budget logic, and fallbacks
6. Returns a structured JSON result with the order outcome

## Architecture

```
Client → FastAPI → Twilio (outbound call) → Restaurant Phone
                      ↕ WebSocket
                Media Stream Handler
                  ↕           ↕
           Deepgram STT    Cartesia TTS
                  ↕
           Call Orchestrator (FSM)
                  ↕
           Groq / Llama 3.3 70B
```

| Component | Role |
|-----------|------|
| FastAPI | HTTP server, Twilio webhooks, WebSocket handler |
| Twilio | Outbound calling, DTMF, bidirectional audio streams |
| Deepgram | Real-time streaming speech-to-text |
| Groq (Llama 3.3 70B) | LLM for contextual conversation generation |
| Cartesia | Text-to-speech synthesis |

## Setup

### Prerequisites

- Python 3.11+
- A Twilio account with a phone number
- API keys for Deepgram, Groq, and Cartesia
- A publicly accessible URL (use ngrok for local dev)

### Installation

```bash
git clone <repo-url>
cd pizza-order-voice-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
CARTESIA_API_KEY=your_cartesia_key
BASE_URL=https://your-public-url.ngrok.io
```

### Running

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

For local development with ngrok:

```bash
ngrok http 8000
```

## Usage

Send a POST request to `/place-order` with the order payload:

```bash
curl -X POST http://localhost:8000/place-order \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Jordan Mitchell",
    "phone_number": "5125550147",
    "delivery_address": "4821 Elm Street, Apt 3B, Austin, TX 78745",
    "pizza": {
      "size": "large",
      "crust": "thin",
      "toppings": ["pepperoni", "mushroom", "green pepper"],
      "acceptable_topping_subs": ["sausage", "bacon", "onion", "spinach", "jalapeño"],
      "no_go_toppings": ["olives", "anchovies", "pineapple"]
    },
    "side": {
      "first_choice": "buffalo wings, 12 count",
      "backup_options": ["garlic bread", "breadsticks", "mozzarella sticks"],
      "if_all_unavailable": "skip"
    },
    "drink": {
      "first_choice": "2L Coke",
      "alternatives": ["2L Pepsi", "2L Sprite"],
      "skip_if_over_budget": true
    },
    "budget_max": 45.00,
    "special_instructions": "Ring doorbell, don't knock (baby sleeping)"
  }'
```

## Order Payload Reference

| Field | Type | Description |
|-------|------|-------------|
| `customer_name` | string | Name for the order |
| `phone_number` | string | 10-digit callback number |
| `delivery_address` | string | Full address including zip code |
| `pizza.size` | string | Pizza size |
| `pizza.crust` | string | Crust type |
| `pizza.toppings` | list | Desired toppings in preference order |
| `pizza.acceptable_topping_subs` | list | Allowed substitute toppings |
| `pizza.no_go_toppings` | list | Toppings to always reject |
| `side.first_choice` | string | Preferred side |
| `side.backup_options` | list | Fallback sides in order |
| `side.if_all_unavailable` | string | `"skip"` to skip if none available |
| `drink.first_choice` | string | Preferred drink |
| `drink.alternatives` | list | Fallback drinks in order |
| `drink.skip_if_over_budget` | bool | Skip drink if it exceeds budget |
| `budget_max` | number | Maximum total in dollars |
| `special_instructions` | string | Delivery notes for the employee |

## Call Flow

### IVR Navigation
The agent handles the restaurant's automated menu deterministically — no LLM involved. It responds in the exact format each prompt expects (DTMF for digits, bare speech for names/zip codes).

### Hold Detection
While on hold, the agent stays silent. It detects a human pickup by recognizing conversational greeting patterns or natural sentences (4+ words that aren't IVR prompts).

### Conversation
The LLM drives the human conversation through ordered phases:

`GREETING → DELIVERY_INFO → PIZZA_ORDER → SIDE_ORDER → DRINK_ORDER → PRICE_COLLECTION → SPECIAL_INSTRUCTIONS → CLOSING`

### Business Rules
- Topping substitutions: only accept from the whitelist, always reject the blacklist
- Side fallbacks: try backup options in order, skip if all unavailable
- Drink budget gating: skip the drink if it would push total over `budget_max`
- Budget enforcement: end the call if pizza + side exceeds budget

## Call Outcomes

| Outcome | Condition |
|---------|-----------|
| `completed` | Full order placed, prices collected, special instructions delivered |
| `nothing_available` | Core pizza item can't be ordered |
| `over_budget` | Pizza + side exceeds budget before drink |
| `detected_as_bot` | Employee suspects the agent is a bot |

## Example Output

```json
{
  "outcome": "completed",
  "pizza": {
    "description": "large thin with pepperoni, onion, green pepper",
    "substitutions": {"mushroom": "onion"},
    "price": 18.50
  },
  "side": {
    "description": "garlic bread",
    "original": "buffalo wings, 12 count",
    "price": 6.99
  },
  "drink": {
    "description": "2L Coke",
    "price": 3.49
  },
  "total": 28.98,
  "delivery_time": "35 minutes",
  "order_number": "4412",
  "special_instructions_delivered": true
}
```

## Testing

1. Point the agent at your own phone number
2. Pick up and play the restaurant employee
3. Walk through the IVR prompts, then have a natural conversation
4. Check the JSON output matches what happened on the call

## License

MIT
