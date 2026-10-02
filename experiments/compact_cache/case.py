"""The same short conversation for every Harness adapter."""

MARKER = "BLUE-ORCHID-731"
FIRST = f"Turn 1: Remember {MARKER}. Reply only ACK. Do not use tools."
CONTEXT = ("Turn 2: Continue this same conversation. "
           + "Context item. " * 120 + "Reply only ACK. Do not use tools.")
MORE_CONTEXT = ("Turn 3: Continue this same conversation. "
                + "Context item. " * 120 + "Reply only ACK. Do not use tools.")
SECOND = "Turn 2: What marker did I ask you to remember? Reply with it only. Do not use tools."
THIRD = "Turn 3: What marker did I ask you to remember? Reply with it only. Do not use tools."
FOURTH = "Turn 4: What marker did I ask you to remember? Reply with it only. Do not use tools."
