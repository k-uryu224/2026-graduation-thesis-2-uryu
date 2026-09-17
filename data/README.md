# Data

ローカルで生成・取得した dataset を置く場所です。大容量データは Git 管理外です。

データそのものとは別に、生成 config、dataset ID、件数、checksum、保存先を
`experiments/` へ記録し、後から同じデータを再生成・特定できるようにします。
