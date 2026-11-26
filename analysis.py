"""Analyze the included synthetic sales database and create a report and chart."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "sales.db"
CHART_PATH = ROOT / "sales_revenue.png"
REPORT_PATH = ROOT / "day3_report.md"
DEMO_PRICES = [
    120, 150, 100, 230, 180, 160, 140, 210, 170, 190,
    130, 220, 160, 150, 180, 200, 140, 170, 230, 160,
    190, 120, 210, 180, 150, 200, 170, 140, 220, 9999,
]
REQUIRED_FIELDS = ("sale_id", "sale_date", "product", "quantity", "unit_price")


def create_demo_database(db_path: Path) -> None:
    """Create a reproducible demo database without overwriting an existing file."""
    if db_path.exists():
        raise FileExistsError(
            f"Database already exists: {db_path}. It will not be overwritten."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    start_date = date(2026, 1, 1)
    records = []
    for index, price in enumerate(DEMO_PRICES):
        quantity = 1 + (index * 7 % 5)
        if price == 9999:
            quantity = 1
        records.append(
            (
                index + 1,
                (start_date + timedelta(days=index)).isoformat(),
                f"Product {(index % 4) + 1}",
                quantity,
                price,
            )
        )

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE sales (
                sale_id INTEGER PRIMARY KEY,
                sale_date TEXT NOT NULL
                    CHECK (date(sale_date) IS NOT NULL),
                product TEXT NOT NULL CHECK (length(trim(product)) > 0),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL CHECK (unit_price > 0)
            );
            CREATE TABLE dataset_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO sales (sale_id, sale_date, product, quantity, unit_price)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.executemany(
            "INSERT INTO dataset_metadata (key, value) VALUES (?, ?)",
            [
                ("source", "synthetic demonstration data"),
                ("currency", "unspecified units"),
                ("outlier_note", "one intentionally extreme unit price for analysis"),
            ],
        )


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile (same convention as NumPy)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile for an empty list.")
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * weight


def load_and_validate(db_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not db_path.exists():
        create_demo_database(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(sales)").fetchall()
        }
        missing_columns = set(REQUIRED_FIELDS) - columns
        if missing_columns:
            raise ValueError(
                f"Database table 'sales' is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        rows = connection.execute(
            """
            SELECT sale_id, sale_date, product, quantity, unit_price
            FROM sales
            ORDER BY sale_date, sale_id
            """
        ).fetchall()
        if not rows:
            raise ValueError("The sales table is empty.")

        null_rows = connection.execute(
            """
            SELECT COUNT(*) FROM sales
            WHERE sale_id IS NULL OR sale_date IS NULL OR product IS NULL
               OR quantity IS NULL OR unit_price IS NULL
            """
        ).fetchone()[0]
        duplicate_ids = connection.execute(
            """
            SELECT COUNT(*) - COUNT(DISTINCT sale_id)
            FROM sales
            """
        ).fetchone()[0]
        duplicate_records = connection.execute(
            """
            SELECT COALESCE(SUM(duplicate_count - 1), 0) FROM (
                SELECT COUNT(*) AS duplicate_count
                FROM sales
                GROUP BY sale_date, product, quantity, unit_price
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM dataset_metadata"
            ).fetchall()
        )

    sales = []
    for row in rows:
        sale_date = date.fromisoformat(row["sale_date"])
        quantity = row["quantity"]
        unit_price = row["unit_price"]
        if quantity <= 0 or unit_price <= 0:
            raise ValueError(f"Invalid quantity or price in sale {row['sale_id']}.")
        sales.append(
            {
                "sale_id": row["sale_id"],
                "sale_date": sale_date,
                "product": row["product"],
                "quantity": quantity,
                "unit_price": float(unit_price),
                "revenue": quantity * float(unit_price),
            }
        )

    quality = {
        "integrity": integrity,
        "rows": len(sales),
        "null_rows": null_rows,
        "duplicate_ids": duplicate_ids,
        "duplicate_records": duplicate_records,
        "metadata": metadata,
    }
    return sales, quality


def analyze(sales: list[dict[str, Any]], quality: dict[str, Any]) -> dict[str, Any]:
    prices = [sale["unit_price"] for sale in sales]
    q1 = percentile(prices, 0.25)
    q3 = percentile(prices, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = [
        sale
        for sale in sales
        if sale["unit_price"] < lower_fence or sale["unit_price"] > upper_fence
    ]
    outlier_ids = {sale["sale_id"] for sale in outliers}
    regular_sales = [sale for sale in sales if sale["sale_id"] not in outlier_ids]
    regular_prices = [sale["unit_price"] for sale in regular_sales]

    revenue_by_date: dict[date, float] = defaultdict(float)
    for sale in sales:
        revenue_by_date[sale["sale_date"]] += sale["revenue"]

    total_revenue = sum(sale["revenue"] for sale in sales)
    regular_revenue = sum(sale["revenue"] for sale in regular_sales)
    outlier_revenue = total_revenue - regular_revenue
    highest_date, highest_revenue = max(revenue_by_date.items(), key=lambda item: item[1])
    lowest_date, lowest_revenue = min(revenue_by_date.items(), key=lambda item: item[1])

    return {
        "quality": quality,
        "count": len(sales),
        "mean_price": sum(prices) / len(prices),
        "median_price": percentile(prices, 0.5),
        "mean_price_without_outliers": sum(regular_prices) / len(regular_prices),
        "median_price_without_outliers": percentile(regular_prices, 0.5),
        "q1": q1,
        "q3": q3,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers": outliers,
        "revenue_by_date": dict(sorted(revenue_by_date.items())),
        "total_revenue": total_revenue,
        "regular_revenue": regular_revenue,
        "outlier_revenue": outlier_revenue,
        "outlier_revenue_share": (
            outlier_revenue / total_revenue * 100 if total_revenue else 0
        ),
        "highest_date": highest_date,
        "highest_revenue": highest_revenue,
        "lowest_date": lowest_date,
        "lowest_revenue": lowest_revenue,
    }


def create_chart(results: dict[str, Any], chart_path: Path) -> None:
    dates = list(results["revenue_by_date"])
    revenues = list(results["revenue_by_date"].values())
    outlier_dates = {
        sale["sale_date"] for sale in results["outliers"]
    }
    colors = ["#d9534f" if day in outlier_dates else "#4c78a8" for day in dates]

    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar([day.strftime("%d %b") for day in dates], revenues, color=colors)
    axis.set_title("Выручка по датам — синтетические демонстрационные данные")
    axis.set_xlabel("Дата")
    axis.set_ylabel("Выручка, условные единицы")
    axis.tick_params(axis="x", labelrotation=55)
    axis.grid(axis="y", alpha=0.25)
    if results["outliers"]:
        outlier_date = results["outliers"][0]["sale_date"]
        axis.annotate(
            "Строка с выбросом",
            xy=(
                dates.index(outlier_date),
                results["revenue_by_date"][outlier_date],
            ),
            xytext=(0, -30),
            textcoords="offset points",
            ha="center",
            color="#9d2622",
            arrowprops={"arrowstyle": "->", "color": "#9d2622"},
        )
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)


def create_report(results: dict[str, Any], report_path: Path, chart_path: Path) -> None:
    quality = results["quality"]
    outlier_summary = (
        "; ".join(
            f"продажа №{sale['sale_id']} от {sale['sale_date'].isoformat()}: "
            f"{sale['unit_price']:,.2f}"
            for sale in results["outliers"]
        )
        .replace(",", " ")
        or "не обнаружены"
    )
    mean_change = (
        results["mean_price"] / results["mean_price_without_outliers"] - 1
    ) * 100
    revenue_share = results["outlier_revenue_share"]
    quality_status = (
        "пройдены"
        if not quality["null_rows"]
        and not quality["duplicate_ids"]
        and not quality["duplicate_records"]
        and quality["integrity"] == "ok"
        else "есть проблемы"
    )

    conclusions = [
        (
            f"В базе {results['count']} строк; проверка целостности SQLite: "
            f"`{quality['integrity']}`, пропусков: {quality['null_rows']}, "
            f"повторных ID: {quality['duplicate_ids']}, повторных записей: "
            f"{quality['duplicate_records']}. Проверки качества {quality_status}."
        ),
        (
            f"Средняя цена равна {results['mean_price']:,.2f}, медиана — "
            f"{results['median_price']:,.2f} условных единиц. Разница показывает, "
            "что распределение цен чувствительно к крайнему значению."
        ),
        (
            f"Метод IQR выделил {len(results['outliers'])} выброс(а): "
            f"{outlier_summary}. Верхняя граница без выбросов — "
            f"{results['upper_fence']:,.2f}."
        ),
        (
            f"После исключения отмеченных выбросов средняя цена составляет "
            f"{results['mean_price_without_outliers']:,.2f}, медиана — "
            f"{results['median_price_without_outliers']:,.2f}. Среднее с выбросом "
            f"выше базового среднего на {mean_change:,.1f}%."
        ),
        (
            f"Суммарная выручка с выбросом — {results['total_revenue']:,.2f}; "
            f"без него — {results['regular_revenue']:,.2f}. Выброс формирует "
            f"{revenue_share:,.1f}% выручки в этом демонстрационном наборе."
        ),
        (
            f"Максимальная дневная выручка зафиксирована {results['highest_date']}: "
            f"{results['highest_revenue']:,.2f}; минимальная — "
            f"{results['lowest_date']}: {results['lowest_revenue']:,.2f}. "
            "Пик совпадает с датой искусственно добавленного выброса."
        ),
    ]

    report = f"""# День 3 — исследование данных и поиск аномалий

> **Важно:** это воспроизводимый учебный набор синтетических данных, а не реальные продажи. Одно экстремальное значение цены намеренно добавлено, чтобы показать влияние выброса.

## Качество данных

- Источник: `{quality['metadata'].get('source', 'не указан')}`.
- Строк в таблице `sales`: {quality['rows']}.
- Целостность SQLite (`PRAGMA integrity_check`): `{quality['integrity']}`.
- Строк с пропусками в обязательных полях: {quality['null_rows']}.
- Повторных идентификаторов: {quality['duplicate_ids']}.
- Повторных записей (дата, товар, количество, цена): {quality['duplicate_records']}.
- Проверки ограничений схемы: положительное количество и цена, непустое название товара, корректная дата.

## Статистика и выбросы

Цены указаны в условных единицах, поскольку валюта в учебных данных не задана.

| Показатель | Все записи | Без выявленных выбросов |
|---|---:|---:|
| Количество продаж | {results['count']} | {results['count'] - len(results['outliers'])} |
| Средняя цена | {results['mean_price']:,.2f} | {results['mean_price_without_outliers']:,.2f} |
| Медианная цена | {results['median_price']:,.2f} | {results['median_price_without_outliers']:,.2f} |
| Выручка | {results['total_revenue']:,.2f} | {results['regular_revenue']:,.2f} |

Для выявления выбросов применён метод межквартильного размаха: `IQR = Q3 − Q1`; границы — `Q1 − 1.5 × IQR` и `Q3 + 1.5 × IQR`. Полученные значения: `Q1 = {results['q1']:,.2f}`, `Q3 = {results['q3']:,.2f}`, нижняя граница `{results['lower_fence']:,.2f}`, верхняя граница `{results['upper_fence']:,.2f}`. Выбросы по цене: {outlier_summary}.

### Почему необычные значения искажают анализ

Среднее арифметическое использует величину каждого наблюдения, поэтому одна очень высокая цена заметно поднимает среднюю цену. Если такая продажа включена в дневную выручку, она также увеличивает сумму и визуально создаёт пик на графике. Медиана зависит от порядка значений, а не от размера крайних значений, поэтому обычно устойчивее. Удалять необычную запись автоматически нельзя: сначала следует выяснить, является ли она ошибкой ввода или реальной редкой продажей. Здесь значение намеренно синтетическое и сохраняется в базе для демонстрации; сравнение без него приведено отдельно.

## Выручка по датам

![Динамика выручки по датам](sales_revenue.png)

Красным отмечен день с продажей, цена которой признана выбросом методом IQR.

## Выводы

{chr(10).join(f'{index}. {item}' for index, item in enumerate(conclusions, start=1))}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверить базу продаж, исследовать выбросы и создать отчёт с графиком."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="путь к SQLite базе (по умолчанию sales.db рядом со скриптом)",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=CHART_PATH,
        help="путь к PNG графику (по умолчанию sales_revenue.png)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="путь к Markdown отчёту (по умолчанию day3_report.md)",
    )
    args = parser.parse_args()

    sales, quality = load_and_validate(args.db)
    results = analyze(sales, quality)
    create_chart(results, args.chart)
    create_report(results, args.report, args.chart)

    print(f"База: {args.db}")
    print(f"Строк: {results['count']}; пропуски: {quality['null_rows']}")
    print(
        f"Средняя цена: {results['mean_price']:.2f}; "
        f"медианная цена: {results['median_price']:.2f}"
    )
    print(f"Обнаружено выбросов по цене: {len(results['outliers'])}")
    print(f"График: {args.chart}")
    print(f"Отчёт: {args.report}")


if __name__ == "__main__":
    main()
