import math

TICK_VALUE = 12.5  # Valeur d'un tick en dollars

def calculate_pnl(num_trades, win_ticks, loss_ticks, win_rate):
    wins = num_trades * win_rate
    losses = num_trades - wins
    pnl = (wins * win_ticks - losses * loss_ticks) * TICK_VALUE
    return pnl, wins, losses

def calculate_breakeven_point(win_ticks, loss_ticks):
    return loss_ticks / (win_ticks + loss_ticks)

def main():
    num_trades = 3091
    win_ticks = 6  # Nombre de ticks gagnés sur un trade gagnant
    loss_ticks = 4.5  # Nombre de ticks perdus sur un trade perdant
    win_rates = [i/100 for i in range(40, 101, 5)]  # De 50% à 100% par pas de 5%

    breakeven_point = calculate_breakeven_point(win_ticks, loss_ticks)

    print("Paramètres de simulation :")
    print(f"Nombre de trades : {num_trades}")
    print(f"Valeur d'un tick : ${TICK_VALUE}")
    print(f"Gain par trade gagnant : {win_ticks} tick(s) (${win_ticks * TICK_VALUE})")
    print(f"Perte par trade perdant : {loss_ticks} tick(s) (${loss_ticks * TICK_VALUE})")
    print(f"Ratio de trades gagnants/perdants à l'équilibre : {1/(1-breakeven_point):.2f}")
    print(f"Point d'équilibre (breakeven) : {breakeven_point:.2%}")
    print("\nRésultats de la simulation :")
    print("Winrate | Trades gagnants | Trades perdants | PnL ($)")
    print("-" * 70)

    breakeven_inserted = False

    for win_rate in win_rates:
        if not breakeven_inserted and win_rate > breakeven_point:
            exact_pnl, exact_wins, exact_losses = calculate_pnl(num_trades, win_ticks, loss_ticks, breakeven_point)
            pnl_str = f"{exact_pnl:10.2f}" if abs(exact_pnl) >= 0.01 else "     0.00"
            print(f"\033[92m{breakeven_point*100:6.1f}% | {exact_wins:16.2f} | {exact_losses:15.2f} | {pnl_str}  (Break Even Théorique)\033[0m")
            breakeven_inserted = True

        pnl, wins, losses = calculate_pnl(num_trades, win_ticks, loss_ticks, win_rate)
        print(f"{win_rate*100:6.1f}% | {wins:16.2f} | {losses:15.2f} | {pnl:10.2f}")

if __name__ == "__main__":
    main()