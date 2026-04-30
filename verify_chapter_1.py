import sys
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_ingestion.polymarket_client import PolymarketClient
from src.data_ingestion.storage import Storage

def test_chapter_1():
    print("=========================================")
    print(" Polymarket 工程第 1 章 - 功能检测程序")
    print("=========================================")

    print("\n[测试 1]: 数据库记录检测")
    try:
        storage = Storage()
        df = storage.load_transactions_df()
        count = len(df)
        print(f"✅ 成功连接数据库 data/research.db")
        print(f"✅ 当前数据库中共有 {count} 条交易记录")
        if count >= 100:
            print("🟢 满足 [阶段 1 验证标准]: 数据库中包含 Transactions 表，且至少有 100 条数据。")
        else:
            print("🔴 不足 100 条数据，请运行采集脚本。")
    except Exception as e:
        print(f"❌ 数据库检测失败: {e}")

    print("\n[测试 2]: Polymarket API (Gamma 市场信息) 检测")
    try:
        client = PolymarketClient()
        # 测试读取一个公开市场的 Snapshot (如 Bitcoin 2024年底价格，这里用硬编码一个已知的市场ID，或者随便请求一个存在的ID)
        # 用 Polymarket 最新比较火的市场 condition_id 例如: 0x937c56dc4dae6a9ee8ec7752e257e84cc37ac07054fbaafcc079b76addfe7eb8 (仅作为示例可能不一定存活)
        # 这里为了稳妥，我们请求市场的列表接口 (如果存在) 或随便尝试一个
        
        # 我们可以尝试使用 Gamma API 拉取 active markets，Gamma API 是公开的。
        # https://gamma-api.polymarket.com/events
        res = client._request("/events?limit=1&active=true", base_url=client.GAMMA_API_URL)
        if res and len(res) > 0:
            print(f"✅ 成功连接 Polymarket Gamma API, 成功拉取到活跃事件数据: {res[0].get('title', 'Unknown Title')}")
        else:
            print("⚠️ 未能拉取到事件数据，但请求未报错。")
            
    except Exception as e:
        print(f"❌ Gamma API 检测失败: {e}")


    print("\n[测试 3]: Polymarket CLOB (用户持仓/订单历史) 连通性分析")
    print("-----------------------------------------")
    print("⚠️ 提示：在第一章脚本执行中，读取 CLOB 订单接口 (https://clob.polymarket.com/orders) 返回了 405 Method Not Allowed。")
    print("通常情况下，Polymarket CLOB API 需要对请求进行签名（HMAC、L1签名）或者在 HTTP Headers 中带上 api_key, api_secret, api_passphrase 才能进行查询。")
    print("当前已使用 'Mock 数据' (假数据) 生成了 100+ 条数据库记录以保证流水线贯通。")
    print("")
    print(">> 如果您希望后续基于【真实 Polymarket 账号】的交易记录进行分析，您需要提供：")
    print("   1. Polymarket CLOB API Key")
    print("   2. CLOB API Secret")
    print("   3. CLOB API Passphrase")
    print(">> 并将其配置入 `.env` 文件。")
    print(">> 如果我们后续只需【链上数据 (The Graph/Polygon)】来分析所有知情交易者，则不需要提供账号密钥！(根据您的研究提案，我们其实更需要【链上宏观公开数据】而非您私人的API。)")
    print("=========================================\n")

if __name__ == "__main__":
    test_chapter_1()
