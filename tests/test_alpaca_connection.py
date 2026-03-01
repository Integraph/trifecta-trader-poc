"""Test Alpaca paper trading connection.

Run with: pytest tests/test_alpaca_connection.py -v
Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars (or in .env).
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("APCA_API_KEY_ID"),
    reason="Alpaca API credentials not set (APCA_API_KEY_ID missing)",
)


class TestAlpacaConnection:

    def test_paper_account_accessible(self):
        """Should connect to Alpaca paper trading and get account info."""
        from alpaca.trading.client import TradingClient

        client = TradingClient(
            api_key=os.environ["APCA_API_KEY_ID"],
            secret_key=os.environ["APCA_API_SECRET_KEY"],
            paper=True,
        )
        account = client.get_account()

        assert account is not None
        assert float(account.portfolio_value) > 0
        print(f"\nAlpaca Paper Account:")
        print(f"  Portfolio value: ${float(account.portfolio_value):,.2f}")
        print(f"  Buying power:    ${float(account.buying_power):,.2f}")
        print(f"  Cash:            ${float(account.cash):,.2f}")
        print(f"  Status:          {account.status}")

    def test_position_manager_integration(self):
        """PositionManager should work with real Alpaca client."""
        from alpaca.trading.client import TradingClient
        from src.execution.position_manager import PositionManager

        client = TradingClient(
            api_key=os.environ["APCA_API_KEY_ID"],
            secret_key=os.environ["APCA_API_SECRET_KEY"],
            paper=True,
        )
        pm = PositionManager(client)
        state = pm.get_account_state()

        assert state.portfolio_value > 0
        assert state.buying_power >= 0

        positions = pm.get_positions()
        print(f"\n  Open positions: {len(positions)}")
        for ticker, pos in positions.items():
            print(f"    {ticker}: {pos.qty} shares @ ${pos.current_price:.2f}")

    def test_executor_initialization(self):
        """TradeExecutor should initialize with paper=True."""
        from src.execution.executor import TradeExecutor

        executor = TradeExecutor()
        # Verify the client was created successfully
        assert executor.client is not None

        # Verify we can get account (proves connection works)
        account = executor.client.get_account()
        assert account is not None
        print(f"\n  Executor connected to paper account")
        print(f"  Account status: {account.status}")
