"""
TauBenchBackend：评测模式的后端。每个方法都转成 τ-bench 的 env.step(Action(...))。

为什么评测不用 MySQLBackend：
- τ-bench 的评分方式是比对"数据库最终状态的哈希值"和标准答案，必须由 τ-bench 自己的工具修改它自己的数据；
- 每一次工具调用（包括 think、calculate）都要记进 env 的动作日志。

安装方法见 eval/README.md。

评测时 Agent 的工具通过 call(name, **kwargs) 调用：参数原样交给 env.step，和 τ-bench 自带的单 Agent 基线完全一样——
参数错误、缺字段等情况都由 τ-bench 的工具自己报错，两种 Agent 受到的待遇相同，比较才公平。
"""

from typing import Any

from backends.base import AirlineBackend


class TauBenchBackend(AirlineBackend):
    def __init__(self, env: Any):
        """env: tau_bench.envs.airline.MockAirlineDomainEnv 实例（已 reset 到某个任务）。"""
        self.env = env
        self.last_response = None          # 最近一次 env.step 的完整返回（含 reward / done），评测循环会用到
        self.actions: list[dict] = []      # Agent 实际调用过的工具（失败分析用；env.actions 在算分时会混入标准答案）

    def _step(self, name: str, **kwargs) -> str:
        from tau_bench.types import Action   # 延迟导入：产品模式不需要安装 tau-bench

        self.actions.append({"name": name, "kwargs": kwargs})
        self.last_response = self.env.step(Action(name=name, kwargs=kwargs))
        return self.last_response.observation

    def call(self, name: str, **kwargs) -> str:
        """按工具名调用（评测时 Agent 的工具走这里）。"""
        return self._step(name, **kwargs)

    @property
    def done(self) -> bool:
        """转人工（transfer_to_human_agents）会让 τ-bench 直接结束本题并打分。"""
        return bool(self.last_response is not None and self.last_response.done)

    def get_user_details(self, user_id):
        return self._step("get_user_details", user_id=user_id)

    def get_reservation_details(self, reservation_id):
        return self._step("get_reservation_details", reservation_id=reservation_id)

    def list_all_airports(self):
        return self._step("list_all_airports")

    def search_direct_flight(self, origin, destination, date):
        return self._step("search_direct_flight", origin=origin, destination=destination, date=date)

    def search_onestop_flight(self, origin, destination, date):
        return self._step("search_onestop_flight", origin=origin, destination=destination, date=date)

    def book_reservation(self, user_id, origin, destination, flight_type, cabin, flights, passengers,
                         payment_methods, total_baggages, nonfree_baggages, insurance):
        return self._step("book_reservation", user_id=user_id, origin=origin, destination=destination,
                          flight_type=flight_type, cabin=cabin, flights=flights, passengers=passengers,
                          payment_methods=payment_methods, total_baggages=total_baggages,
                          nonfree_baggages=nonfree_baggages, insurance=insurance)

    def update_reservation_flights(self, reservation_id, cabin, flights, payment_id):
        return self._step("update_reservation_flights", reservation_id=reservation_id, cabin=cabin,
                          flights=flights, payment_id=payment_id)

    def update_reservation_passengers(self, reservation_id, passengers):
        return self._step("update_reservation_passengers", reservation_id=reservation_id, passengers=passengers)

    def update_reservation_baggages(self, reservation_id, total_baggages, nonfree_baggages, payment_id):
        return self._step("update_reservation_baggages", reservation_id=reservation_id,
                          total_baggages=total_baggages, nonfree_baggages=nonfree_baggages,
                          payment_id=payment_id)

    def cancel_reservation(self, reservation_id):
        return self._step("cancel_reservation", reservation_id=reservation_id)

    def send_certificate(self, user_id, amount):
        return self._step("send_certificate", user_id=user_id, amount=amount)

    # 评测时 think / calculate / transfer 也必须走 env.step（覆盖基类的本地实现）
    def calculate(self, expression):
        return self._step("calculate", expression=expression)

    def think(self, thought):
        return self._step("think", thought=thought)

    def transfer_to_human_agents(self, summary):
        return self._step("transfer_to_human_agents", summary=summary)
