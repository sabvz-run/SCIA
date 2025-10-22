import java.util.Scanner;

public class JogoDaVelha {
    static char[][] tabuleiro = new char[3][3];
    static Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        inicializarTabuleiro();
        char jogadorAtual = 'X';
        boolean jogoAcabou = false;

        while (!jogoAcabou) {
            exibirTabuleiro();

            System.out.println("Jogador " + jogadorAtual + ", escolha sua posição (1-9): ");
            int posicao = scanner.nextInt();

            if (posicaoValida(posicao) && !estaOcupada(posicao)) {
                marcarPosicao(posicao, jogadorAtual);

                if (verificarVitoria(jogadorAtual)) {
                    exibirTabuleiro();
                    System.out.println("Jogador " + jogadorAtual + " venceu!");
                    jogoAcabou = true;
                } else if (tabuleiroCheio()) {
                    exibirTabuleiro();
                    System.out.println("Empate!");
                    jogoAcabou = true;
                } else {
                    jogadorAtual = (jogadorAtual == 'X') ? 'O' : 'X';
                }
            } else {
                System.out.println("Posição inválida ou ocupada!");
            }
        }

        scanner.close();
    }

    static void inicializarTabuleiro() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tabuleiro[i][j] = ' ';
            }
        }
    }

    static void exibirTabuleiro() {
        System.out.println("\n 1 | 2 | 3");
        System.out.println("-----------");
        for (int i = 0; i < 3; i++) {
            System.out.print(" " + tabuleiro[i][0] + " | " + tabuleiro[i][1] + " | " + tabuleiro[i][2]);
            System.out.println();
            if (i < 2) System.out.println("-----------");
        }
        System.out.println();
    }

    static boolean posicaoValida(int pos) {
        return pos >= 1 && pos <= 9;
    }

    static boolean estaOcupada(int pos) {
        int linha = (pos - 1) / 3;
        int coluna = (pos - 1) % 3;
        return tabuleiro[linha][coluna] != ' ';
    }

    static void marcarPosicao(int pos, char jogador) {
        int linha = (pos - 1) / 3;
        int coluna = (pos - 1) % 3;
        tabuleiro[linha][coluna] = jogador;
    }

    static boolean verificarVitoria(char jogador) {
        for (int i = 0; i < 3; i++) {
            if (tabuleiro[i][0] == jogador && tabuleiro[i][1] == jogador && tabuleiro[i][2] == jogador) {
                return true;
            }
        }

        for (int j = 0; j < 3; j++) {
            if (tabuleiro[0][j] == jogador && tabuleiro[1][j] == jogador && tabuleiro[2][j] == jogador) {
                return true;
            }
        }

        if (tabuleiro[0][0] == jogador && tabuleiro[1][1] == jogador && tabuleiro[2][2] == jogador) {
            return true;
        }

        if (tabuleiro[0][2] == jogador && tabuleiro[1][1] == jogador && tabuleiro[2][0] == jogador) {
            return true;
        }

        return false;
    }

    static boolean tabuleiroCheio() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (tabuleiro[i][j] == ' ') {
                    return false;
                }
            }
        }
        return true;
    }
}
