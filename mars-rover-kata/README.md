# 🚀 Mars Rover Kata — Implementação Orientada a Objetos

Este repositório contém minha solução completa para o **Mars Rover Kata**, utilizando princípios sólidos de Engenharia de Software:

- Padrão **State** (cada direção é um objeto)
- **Polimorfismo** para eliminar condicionais
- **Value Objects** imutáveis (Position, Plateau)
- **Factory Pattern**
- Código limpo, sustentável e extensível
- Testes automatizados com Pytest

---

## 🧩 Sobre o Problema

O enunciado completo está em: **problem.md**

---

## 📂 Estrutura do Projeto

```
mars-rover-kata/
├── README.md
├── problem.md
├── requirements.txt
├── .gitignore
├── main.py
├── mars_rover/
│ ├── init.py
│ ├── entities.py
│ ├── directions.py
│ ├── factory.py
│ └── rover.py
└── tests/
├── init.py
├── test_rover_basic.py
└── test_rover_commands.py
```


---

## ▶ Como Executar

```bash
python main.py

pytest -q


🛠️ Extensões Futuras

Obstáculos (Rocks)

Direções diagonais (NE, NW…)

Múltiplos rovers com detecção de colisão

Parser de input estilo NASA


---

# ✅ **problem.md** (enunciado oficial)

```markdown
# Mars Rover Kata — Problem Statement

A squad of robotic rovers are to be landed by NASA on a plateau on Mars.  
This plateau, which is curiously rectangular, must be navigated by the rovers so that their on-board cameras can get a complete view of the surrounding terrain.

A rover's position and location is represented by a combination of x and y coordinates and a letter representing one of the four cardinal compass points.

The plateau is divided into a grid. Coordinates are **0,0** in the bottom-left.

The rover receives a list of commands:

- **L**: rotate 90º left
- **R**: rotate 90º right
- **M**: move forward one grid point

Rovers cannot leave the plateau.

## Input Example

5 5
1 2 N
LMLMLMLMM
3 3 E
MMRMMRMRRM


## Output Expected

1 3 N
5 1 E
