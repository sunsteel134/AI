import random

class Belief:
    def __init__(self, name, is_religion=False):
        self.name = name
        self.is_religion = is_religion
        self.strength = random.uniform(0.5, 1.0)

belief_options = [
    Belief("Cooperation is key"),
    Belief("Individual growth matters most"),
    Belief("Strength through conquest"),
    Belief("Knowledge is power"),
]

class Government:
    def __init__(self, name):
        self.name = name
        self.type = random.choice(["Democracy", "Oligarchy", "Republic"])
        self.tax_rate = random.uniform(0.1, 0.3)
        self.relationships = {}
        self.at_war_with = set()
        self.treasury = 0
        self.council = []
        self.council_size = random.randint(3, 7)
        self.government = None

    def form_council(self, agents):
        self.council = random.sample(agents, min(self.council_size, len(agents)))
        self.leader = random.choice(self.council)  # Choose a leader from the council

    def collect_taxes(self, agents):
        for agent in agents:
            tax = int(agent.resources * self.tax_rate)
            agent.resources -= tax
            self.treasury += tax

    def set_relationship(self, other_gov, value):
        self.relationships[other_gov] = value

    def declare_war(self, other_gov):
        self.at_war_with.add(other_gov)
        other_gov.at_war_with.add(self)

    def make_peace(self, other_gov):
        self.at_war_with.discard(other_gov)
        other_gov.at_war_with.discard(self)

    def send_aid(self, other_gov, amount):
        if self.treasury >= amount:
            self.treasury -= amount
            other_gov.treasury += amount
            self.relationships[other_gov] = min(1.0, self.relationships.get(other_gov, 0) + 0.1)

    def raise_tax(self, amount):
        self.tax_rate = min(1.0, self.tax_rate + amount)


class Environment:
    def __init__(self):
        self.resources = 100000
        self.danger_level = 0.1

    def update(self):
        self.resources += random.randint(0, 100)
        self.danger_level = max(0, min(1, self.danger_level + random.uniform(-0.1, 0.1)))
        if self.resources <= 0:
            self.resources = 0

class AIAgent:
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.health = 100
        self.resources = 50
        self.belief = random.choice(belief_options)
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.combat_skill = random.uniform(0.5, 1.0)
        self.leadership = random.uniform(0.5, 1.0)
        self.government = None

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        best_next_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
            return q_values.index(max(q_values))

    def take_action(self, environment, other_agents):
        state = (self.health, self.resources, environment.danger_level)
        action = self.choose_action(state)
        actions = ["gather", "rest", "share", "explore", "fight", "campaign"]
        chosen_action = actions[action]

        if chosen_action == "gather":
            gathered = random.randint(5, 20)
            self.resources += gathered
            environment.resources -= gathered
            reward = gathered
        elif chosen_action == "rest":
            self.health = min(100, self.health + 10)
            reward = 5 if self.health < 50 else 0
        elif chosen_action == "share":
            if self.resources > 20:
                shared = random.randint(5, 20)
                self.resources -= shared
                random.choice(other_agents).resources += shared
                reward = 10
            else:
                reward = -5
        elif chosen_action == "explore":
            discovered = random.randint(0, 30)
            self.resources += discovered
            environment.resources += discovered
            self.health -= random.randint(5, 15)
            reward = discovered - 10
        elif chosen_action == "fight":
            opponent = random.choice(other_agents)
            if self.combat_skill > opponent.combat_skill:
                loot = min(opponent.resources, 20)
                self.resources += loot
                opponent.resources -= loot
                opponent.health -= random.randint(10, 30)
                reward = loot
            else:
                self.health -= random.randint(10, 30)
                reward = -20
        elif chosen_action == "campaign":
            if self.government.leader == self:
                reward = 5
            elif random.random() < self.leadership:
                self.government.leader = self
                reward = 50
            else:
                reward = -10

        self.resources = max(0, self.resources)
        self.health = max(0, self.health)
        next_state = (self.health, self.resources, environment.danger_level)
        self.update_q_value(state, action, reward, next_state)
        return chosen_action, reward

    def leader_action(self, governments):
        reward = 0
        if self.government.leader == self:
            action = random.choice(["diplomacy", "war", "aid", "raise_tax"])
            other_gov = random.choice([gov for gov in governments if gov != self.government])
            if action == "diplomacy":
                self.government.set_relationship(other_gov,
                                                 min(1.0, self.government.relationships.get(other_gov, 0) + 0.1))
                print(f"{self.government.name} improved diplomatic relations with {other_gov.name}")
                reward = 10
            elif action == "war":
                if other_gov not in self.government.at_war_with:
                    self.government.declare_war(other_gov)
                    print(f"{self.government.name} declared war on {other_gov.name}")
                    reward = -10
                else:
                    self.government.make_peace(other_gov)
                    print(f"{self.government.name} made peace with {other_gov.name}")
                    reward = 30
            elif action == "aid":
                aid_amount = random.randint(10, 50)
                if self.government.treasury >= aid_amount:
                    self.government.send_aid(other_gov, aid_amount)
                    print(f"{self.government.name} sent {aid_amount} aid to {other_gov.name}")
                    reward = 50
                else:
                    print(f"{self.government.name} couldn't send aid due to insufficient funds")
                    reward = -10
            elif action == "raise_tax":
                amount_to_raise = random.uniform(0.01, 0.05)
                self.government.raise_tax(amount_to_raise)
                print(f"{self.government.name} raised taxes by {amount_to_raise:.2f}")
                reward = -10
        return reward


class Civilization:
    def __init__(self, num_agents, num_governments):
        self.environment = Environment()
        self.agents = [AIAgent(f"Agent-{i}", state_size=3, action_size=6) for i in range(num_agents)]
        self.governments = [Government(f"Government-{i}") for i in range(num_governments)]

        for government in self.governments:
            available_agents = [agent for agent in self.agents if agent.government is None]
            government_agents = random.sample(available_agents, min(len(available_agents), random.randint(10, 30)))
            for agent in government_agents:
                agent.government = government
            government.form_council(government_agents)

        for agent in self.agents:
            agent.government = random.choice(self.governments)

        for gov in self.governments:
            for other_gov in self.governments:
                if gov != other_gov:
                    gov.set_relationship(other_gov, random.uniform(-0.5, 0.5))

    def run_simulation(self, num_days):
        for day in range(num_days):
            print(f"\nDay {day + 1}:")
            self.environment.update()

            for government in self.governments:
                government.collect_taxes([agent for agent in self.agents if agent.government == government])
                decision = government.leader.leader_action(self.governments)

            for agent in self.agents:
                other_agents = [a for a in self.agents if a != agent]
                action, reward = agent.take_action(self.environment, other_agents, self.governments)
                print(f"{agent.name} ({agent.government.name}): Action: {action}, Reward: {reward}, Health: {agent.health}, Resources: {agent.resources}")

            print(f"Environment - Resources: {self.environment.resources}, Danger Level: {self.environment.danger_level:.2f}")

            for government in self.governments:
                print(f"Government {government.name} - Leader: {government.leader.name if government.leader else 'None'}, Type: {government.type}, Tax Rate: {government.tax_rate:.2f}, Treasury: {government.treasury}")
                print(f" Relationships: {', '.join([f'{g.name}: {v:.2f}' for g, v in government.relationships.items()])}")
                print(f" At war with: {', '.join([g.name for g in government.at_war_with])}")

            self.agents = [agent for agent in self.agents if agent.health > 0]

            if not self.agents:
                print("All agents have died. Simulation ended.")
                break

            for government in self.governments:
                if random.random() < 0.1:
                    government.type = random.choice(["Democracy", "Dictatorship", "Oligarchy"])
                    print(f"Government {government.name} changed to {government.type}")

if __name__ == "__main__":
    civ = Civilization(num_agents=100, num_governments=3)
    civ.run_simulation(num_days=100)
