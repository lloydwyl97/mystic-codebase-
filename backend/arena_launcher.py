import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import docker
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyArena:
    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.docker_client = docker.from_env()
        self.active_containers = {}
        self.leaderboard_key = "strategy_leaderboard"

    def create_strategy_config(
        self,
        strategy_name: str,
        capital: float,
        timeframe: str = "1h",
        symbols: list[str] = None,
    ) -> dict:
        """Create individual strategy configuration"""
        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

        config = {
            "strategy_name": strategy_name,
            "capital": capital,
            "timeframe": timeframe,
            "symbols": symbols,
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "stop_loss": 0.05,
            "take_profit": 0.15,
            "created_at": time.time(),
        }

        return config

    def launch_strategy_container(self, strategy_name: str, config: dict) -> str:
        """Launch individual Docker container for strategy"""
        try:
            # Create strategy directory
            strategy_dir = f"./agents/{strategy_name}"
            os.makedirs(strategy_dir, exist_ok=True)

            # Save config
            with open(f"{strategy_dir}/config.json", "w") as f:
                json.dump(config, f, indent=2)

            # Launch container
            container_name = f"mystic_arena_{strategy_name}_{int(time.time())}"

            container = self.docker_client.containers.run(
                image="mystic_ai:latest",
                name=container_name,
                volumes={
                    os.path.abspath(strategy_dir): {
                        "bind": "/app/agent",
                        "mode": "rw",
                    },
                    os.path.abspath("./data"): {
                        "bind": "/app/data",
                        "mode": "rw",
                    },
                },
                environment={
                    "STRATEGY_NAME": strategy_name,
                    "REDIS_HOST": "host.docker.internal",
                    "REDIS_PORT": "6379",
                },
                detach=True,
                command=f"python strategy_runner.py {strategy_name}",
            )

            self.active_containers[strategy_name] = container.id
            logger.info(f"Launched strategy container: {container_name}")

            # Initialize leaderboard entry
            self.redis_client.hset(
                self.leaderboard_key,
                strategy_name,
                json.dumps(
                    {
                        "profit": 0.0,
                        "trades": 0,
                        "win_rate": 0.0,
                        "sharpe_ratio": 0.0,
                        "last_update": time.time(),
                    }
                ),
            )

            return container.id

        except Exception as e:
            logger.error(f"Failed to launch strategy {strategy_name}: {e}")
            return None

    def generate_strategy_army(self, base_capital: float = 1000.0) -> list[dict]:
        """Generate 100+ diverse strategies"""
        strategies = []

        # Strategy templates
        strategy_templates = [
            {
                "name": "momentum_",
                "type": "momentum",
                "timeframes": ["5m", "15m", "1h"],
            },
            {
                "name": "mean_reversion_",
                "type": "mean_reversion",
                "timeframes": ["1h", "4h", "1d"],
            },
            {
                "name": "breakout_",
                "type": "breakout",
                "timeframes": ["15m", "1h", "4h"],
            },
            {
                "name": "scalping_",
                "type": "scalping",
                "timeframes": ["1m", "5m"],
            },
            {
                "name": "trend_following_",
                "type": "trend",
                "timeframes": ["1h", "4h", "1d"],
            },
            {
                "name": "volatility_",
                "type": "volatility",
                "timeframes": ["5m", "15m", "1h"],
            },
            {"name": "ai_ml_", "type": "ai_ml", "timeframes": ["1h", "4h"]},
            {
                "name": "sentiment_",
                "type": "sentiment",
                "timeframes": ["15m", "1h"],
            },
            {
                "name": "correlation_",
                "type": "correlation",
                "timeframes": ["1h", "4h"],
            },
            {
                "name": "arbitrage_",
                "type": "arbitrage",
                "timeframes": ["1m", "5m"],
            },
        ]

        symbol_groups = [
            ["BTC/USDT", "ETH/USDT"],
            ["BNB/USDT", "ADA/USDT"],
            ["SOL/USDT", "DOT/USDT"],
            ["AVAX/USDT", "MATIC/USDT"],
            ["LINK/USDT", "UNI/USDT"],
            ["ATOM/USDT", "FTM/USDT"],
            ["NEAR/USDT", "ALGO/USDT"],
            ["XRP/USDT", "LTC/USDT"],
            ["BCH/USDT", "ETC/USDT"],
            ["FIL/USDT", "ICP/USDT"],
        ]

        strategy_id = 1
        for template in strategy_templates:
            for timeframe in template["timeframes"]:
                for symbol_group in symbol_groups:
                    for variant in range(1, 4):  # 3 variants per combination
                        strategy_name = f"{template['name']}{strategy_id:03d}"

                        # Vary capital allocation
                        capital_variation = base_capital * (0.8 + (variant * 0.2))

                        config = self.create_strategy_config(
                            strategy_name=strategy_name,
                            capital=capital_variation,
                            timeframe=timeframe,
                            symbols=symbol_group,
                        )

                        config.update(
                            {
                                "strategy_type": template["type"],
                                "variant": variant,
                                "template_id": strategy_id,
                            }
                        )

                        strategies.append(config)
                        strategy_id += 1

                        if strategy_id > 100:  # Limit to 100 strategies
                            break
                    if strategy_id > 100:
                        break
                if strategy_id > 100:
                    break
            if strategy_id > 100:
                break

        return strategies

    def launch_arena(self, num_strategies: int = 100, base_capital: float = 1000.0):
        """Launch the full strategy arena"""
        logger.info(f"ðŸš€ Launching Strategy Arena with {num_strategies} strategies")

        # Generate strategies
        strategies = self.generate_strategy_army(base_capital)[:num_strategies]

        # Launch containers in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for config in strategies:
                strategy_name = config["strategy_name"]
                future = executor.submit(self.launch_strategy_container, strategy_name, config)
                futures.append(future)
                time.sleep(0.1)  # Small delay to prevent overwhelming Docker

        # Wait for all containers to start
        for future in futures:
            try:
                container_id = future.result()
                if container_id:
                    logger.info("âœ… Strategy container launched successfully")
            except Exception as e:
                logger.error(f"âŒ Failed to launch strategy: {e}")

        logger.info(f"ðŸŽ¯ Arena launched with {len(self.active_containers)} active strategies")
        return len(self.active_containers)

    def get_leaderboard(self, limit: int = 20) -> list[dict]:
        """Get current strategy leaderboard"""
        try:
            leaderboard_data = self.redis_client.hgetall(self.leaderboard_key)
            strategies = []

            for strategy_name, data_str in leaderboard_data.items():
                data = json.loads(data_str)
                data["strategy_name"] = strategy_name
                strategies.append(data)

            # Sort by profit
            strategies.sort(key=lambda x: x["profit"], reverse=True)
            return strategies[:limit]

        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            return []

    def survivor_selection(self, survival_rate: float = 0.2):
        """Select top performing strategies and terminate others"""
        try:
            leaderboard = self.get_leaderboard()
            num_survivors = max(1, int(len(leaderboard) * survival_rate))

            survivors = leaderboard[:num_survivors]
            eliminated = leaderboard[num_survivors:]

            logger.info(
                f"ðŸ† Survivor Selection: {len(survivors)} survivors, {len(eliminated)} eliminated"
            )

            # Terminate eliminated strategies
            for strategy in eliminated:
                strategy_name = strategy["strategy_name"]
                if strategy_name in self.active_containers:
                    try:
                        container = self.docker_client.containers.get(
                            self.active_containers[strategy_name]
                        )
                        container.stop(timeout=10)
                        container.remove()
                        del self.active_containers[strategy_name]
                        logger.info(f"ðŸ’€ Eliminated strategy: {strategy_name}")
                    except Exception as e:
                        logger.error(f"Failed to terminate {strategy_name}: {e}")

            return survivors

        except Exception as e:
            logger.error(f"Survivor selection failed: {e}")
            return []

    def monitor_arena(self, check_interval: int = 60):
        """Monitor arena health and performance"""
        logger.info("ðŸ‘ï¸ Starting arena monitoring...")

        while True:
            try:
                # Check container health
                healthy_containers = 0
                for (
                    strategy_name,
                    container_id,
                ) in self.active_containers.items():
                    try:
                        container = self.docker_client.containers.get(container_id)
                        if container.status == "running":
                            healthy_containers += 1
                        else:
                            logger.warning(
                                f"Container {strategy_name} is not running: {container.status}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to check container {strategy_name}: {e}")

                # Get leaderboard
                leaderboard = self.get_leaderboard(10)

                logger.info(
                    f"ðŸ“Š Arena Status: {healthy_containers}/{len(self.active_containers)} containers healthy"
                )
                if leaderboard:
                    top_strategy = leaderboard[0]
                    logger.info(
                        f"ðŸ¥‡ Top Strategy: {top_strategy['strategy_name']} - Profit: ${top_strategy['profit']:.2f}"
                    )

                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("ðŸ›‘ Arena monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(check_interval)


def main():
    """Main arena launcher"""
    arena = StrategyArena()

    # Launch arena with 100 strategies
    num_launched = arena.launch_arena(num_strategies=100, base_capital=1000.0)

    if num_launched > 0:
        logger.info(f"ðŸŽ¯ Arena successfully launched with {num_launched} strategies")

        # Start monitoring
        arena.monitor_arena()
    else:
        logger.error("âŒ Failed to launch arena")


if __name__ == "__main__":
    main()


