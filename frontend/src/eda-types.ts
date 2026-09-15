import type {Profile} from './types';

export type Distribution={column:string;kind:'numeric'|'categorical';count:number;missing:number;unique:number;outliers:number|null;stats:Record<string,number|null>;bins:{label:string;count:number;low?:number;high?:number;other?:boolean}[]};
export type DashboardData=Profile&{id:string;filename:string;source:string;target:string|null;numeric_columns:string[];relationship_columns:string[];types:{label:string;count:number}[];distributions:Distribution[];missing:{column:string;count:number;percent:number}[];correlations:{columns:string[];values:(number|null)[][]};top_correlations:{left:string;right:string;value:number}[];note:string};
export type Relationship={x:string;y:string;total_rows:number;valid_rows:number;omitted_rows:number;sampled:boolean;sample_rows:number;correlation:number|null;points:{row:number;x:number;y:number}[];bounds:{x:[number,number];y:[number,number]}|null};
