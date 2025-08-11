def generate_token(name="MYSTIC", symbol="MYST", supply=1_000_000):
    print(f"[TOKEN] Generating ERC-20 token '{name}' ({symbol}) with supply {supply}")
    contract_code = f"""
    pragma solidity ^0.8.0;

    contract {symbol} {{
        string public name = "{name}";
        string public symbol = "{symbol}";
        uint8 public decimals = 18;
        uint256 public totalSupply = {supply} * (10 ** uint256(decimals));
        mapping(address => uint256) public balanceOf;

        constructor() {{
            balanceOf[msg.sender] = totalSupply;
        }}
    }}
    """
    with open(f"{symbol}_Token.sol", "w") as f:
        f.write(contract_code)
    print(f"[TOKEN] Contract written to {symbol}_Token.sol")
