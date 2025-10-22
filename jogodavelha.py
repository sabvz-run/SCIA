tabuleiro = {
    'a1': ' ', 'a2': ' ', 'a3': ' ',
    'b1': ' ', 'b2': ' ', 'b3': ' ',
    'c1': ' ', 'c2': ' ', 'c3': ' '
}


def exibir_tabuleiro():
    print("\n   1   2   3")
    print("a  " + tabuleiro['a1'] + " | " + tabuleiro['a2'] + " | " + tabuleiro['a3'])
    print("   ---------")
    print("b  " + tabuleiro['b1'] + " | " + tabuleiro['b2'] + " | " + tabuleiro['b3'])
    print("   ---------")
    print("c  " + tabuleiro['c1'] + " | " + tabuleiro['c2'] + " | " + tabuleiro['c3'])
    print()


def verificar_vitoria(jogador):
    linhas = [
        ['a1', 'a2', 'a3'],
        ['b1', 'b2', 'b3'],
        ['c1', 'c2', 'c3']
    ]

    colunas = [
        ['a1', 'b1', 'c1'],
        ['a2', 'b2', 'c2'],
        ['a3', 'b3', 'c3']
    ]

    diagonais = [
        ['a1', 'b2', 'c3'],
        ['a3', 'b2', 'c1']
    ]

    for combo in linhas + colunas + diagonais:
        if all(tabuleiro[pos] == jogador for pos in combo):
            return True

    return False


def tabuleiro_cheio():
    return all(tabuleiro[pos] != ' ' for pos in tabuleiro)


def posicao_valida(posicao):
    return posicao in tabuleiro and tabuleiro[posicao] == ' '


jogador_atual = '●'
jogo_acabou = False

while not jogo_acabou:
    exibir_tabuleiro()

    simbolo = "●" if jogador_atual == '●' else "X"
    print(f"Jogador {jogador_atual} ({simbolo}), escolha sua posição (a1, a2, a3, b1, b2, b3, c1, c2, c3): ")
    posicao = input().lower().strip()

    if posicao_valida(posicao):
        tabuleiro[posicao] = jogador_atual

        if verificar_vitoria(jogador_atual):
            exibir_tabuleiro()
            print(f"Jogador {jogador_atual} venceu!")
            jogo_acabou = True
        elif tabuleiro_cheio():
            exibir_tabuleiro()
            print("Empate!")
            jogo_acabou = True
        else:
            jogador_atual = 'X' if jogador_atual == '●' else '●'
    else:
        print("Posição inválida ou ocupada! Tente novamente.")