#!/usr/bin/env python3
"""
Raw Protocol Test

Test the Forge daemon protocol directly to debug response handling.
"""

import socket
import time


def test_raw():
    """Test raw socket communication with Forge daemon."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(("localhost", 17171))

    # Create buffered reader/writer
    rfile = sock.makefile("r", buffering=1)
    wfile = sock.makefile("w", buffering=1)

    def send(msg):
        print(f">>> SEND: {msg}")
        wfile.write(msg + "\n")
        wfile.flush()

    def recv():
        line = rfile.readline().rstrip("\n")
        if len(line) > 100:
            print(f"<<< RECV: {line[:100]}...")
        else:
            print(f"<<< RECV: {line}")
        return line

    try:
        # Start game
        send("NEWGAME red_aggro.dck white_weenie.dck -i -q -c 60 -s 12345")

        # Read first few decisions and respond
        for i in range(20):
            line = recv()
            if not line:
                print("Connection closed")
                break

            # Skip non-decision lines
            if line.startswith("GAME_START"):
                print("Game started!")
                continue
            if line.startswith("INFO:"):
                print(f"Info: {line}")
                continue
            if not line.startswith("DECISION:"):
                print(f"Other: {line[:80]}")
                continue

            if line.startswith("DECISION:"):
                import json
                data = json.loads(line[9:])
                dtype = data.get("decision_type")
                player = data.get("player", "?")
                turn = data.get("turn", 0)
                phase = data.get("phase", "?")

                print(f"\n=== Decision {i+1}: {dtype} (Turn {turn}, {phase}, {player}) ===")

                if dtype == "choose_action":
                    actions = data.get("actions", [])
                    print(f"Actions: {len(actions)}")
                    for a in actions[:5]:
                        print(f"  [{a.get('index')}] {a.get('card', 'N/A')}: {a.get('description', '')[:50]}")

                    # Find land or pass
                    response = "-1"
                    for a in actions:
                        if a.get("is_land"):
                            response = str(a.get("index"))
                            print(f"  -> Playing land: {response}")
                            break
                    else:
                        print("  -> Passing: -1")

                    # Small delay before sending
                    time.sleep(0.01)
                    send(response)

                elif dtype == "declare_attackers":
                    print("  -> No attackers: (empty)")
                    send("")

                elif dtype == "declare_blockers":
                    print("  -> No blockers: (empty)")
                    send("")

                else:
                    print("  -> Default: 0")
                    send("0")

            elif line.startswith("GAME_RESULT"):
                print(f"\nGame ended: {line}")
                break
            else:
                print(f"Unknown message: {line}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()


if __name__ == "__main__":
    test_raw()
