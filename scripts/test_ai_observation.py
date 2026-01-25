#!/usr/bin/env python3
"""
Test AI Observation Mode

Run games WITHOUT -i flag (using Forge AI) and observe that they work correctly.
This proves the issue is in interactive mode, not the game engine.
"""

import socket


def test_ai_game():
    """Test a game using Forge AI (no -i flag)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120)
    sock.connect(("localhost", 17171))

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
        # Start game WITHOUT -i flag (let Forge AI play both sides)
        send("NEWGAME red_aggro.dck white_weenie.dck -c 30 -s 12345")

        # Read all output until game ends
        while True:
            line = recv()
            if not line:
                print("Connection closed")
                break
            if line.startswith("GAME_RESULT"):
                print(f"\n=== GAME RESULT ===")
                print(line)
                break
            if line.startswith("GAME_TIMEOUT"):
                print(f"\n=== GAME TIMEOUT ===")
                print(line)
                break

    except socket.timeout:
        print("Timeout - game took too long")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()


if __name__ == "__main__":
    test_ai_game()
