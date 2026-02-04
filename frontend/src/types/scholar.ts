export type Paper = {
  id: string;
  title: string;
  displayTitle: string;
  firstSentence: string;
  abstract: string;     // 新增：摘要
  filename: string;     // 新增：文件名
  field: string;        // 新增：领域/聚类名称
  confidence: number;   // 新增：置信度 (0.0 - 1.0)
  size: number;         // 新增：节点大小
  pos: [number, number, number];
  color: string;
  category: string;
  keywords: string[];   // 确保是字符串数组
  cluster: number;
};

export type Edge = {
  source: string;
  target: string;
  weight: number;
  type?: 'intra' | 'bridge';
};

export type GraphResponse = {
  papers: Paper[];
  edges: Edge[];
};

export type QueryResponse = {
  answer: string;
  cites: string[];
};