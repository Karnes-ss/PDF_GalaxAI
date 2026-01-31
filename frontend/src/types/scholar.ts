export type Paper = {
  id: string;
  title: string;
  displayTitle?: string;
  firstSentence?: string;
  pos: [number, number, number];
  color: string;
  category: string;
  keywords?: string[];
  cluster?: number;
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
