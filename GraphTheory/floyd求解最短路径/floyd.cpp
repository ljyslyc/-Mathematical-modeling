/*****************************************************************
Floyd�㷨:����Ѱ�Ҹ����ļ�Ȩͼ�ж�������·�����㷨��

intro:
    ͨ��һ��ͼ��Ȩֵ�����������ÿ���������·������
    ��ͼ�Ĵ�Ȩ�ڽӾ���A=[a(i,j)] n��n��ʼ���ݹ�ؽ���n�θ��£����ɾ���D(0)=A��
    ��һ����ʽ�����������D(1)������ͬ���ع�ʽ��D(1)�����D(2)���������������ͬ���Ĺ�ʽ��D(n-1)���������D(n)��
    ����D(n)��i��j��Ԫ�ر���i�Ŷ��㵽j�Ŷ�������·�����ȣ���D(n)Ϊͼ�ľ������ͬʱ��������һ����̽ڵ����path����¼���������·����

�㷨����
��a)����ʼ����D[u,v]=A[u,v]
��b)��For k:=1 to n
��������For i:=1 to n
����������For j:=1 to n
������������If D[i,j]>D[i,k]+D[k,j] Then
��������������D[i,j]:=D[i,k]+D[k,j];
��c)���㷨������D��Ϊ���е�Ե����·������

�㷨����
    ��ͼ���ڽӾ���G��ʾ�����������Vi��Vj��·�ɴ��G[i,j]=d��d��ʾ��·�ĳ��ȣ�����G[i,j]=��ֵ��
    ����һ������D������¼����������Ϣ��D[i,j]��ʾ��Vi��Vj��Ҫ�����ĵ㣬��ʼ��D[i,j]=j��
    �Ѹ����������ͼ�У��Ƚϲ���ľ�����ԭ���ľ��룬G[i,j] = min( G[i,j], G[i,k]+G[k,j] )�����G[i,j]��ֵ��С����D[i,j]=k��
    ��G�а���������֮����̵�·����Ϣ������D������������ͨ·������Ϣ��
    ���磬ҪѰ�Ҵ�V5��V1��·��������D������D(5,1)=3��˵����V5��V1����V3��·��Ϊ{V5,V3,V1}�����D(5,3)=3��˵��V5��V3ֱ�����������D(3,1)=1��˵��V3��V1ֱ��������

ʱ�临�Ӷ�
    O(n^3)

��ȱ�����
    Floyd�㷨������APSP(All Pairs Shortest Paths)������ͼЧ����ѣ���Ȩ�����ɸ������㷨����Ч����������ѭ���ṹ���գ����ڳ���ͼ��Ч��Ҫ����ִ��|V|��Dijkstra�㷨��
    �ŵ㣺������⣬����������������ڵ�֮�����̾��룬�����д�򵥣�
    ȱ�㣺ʱ�临�ӶȱȽϸߣ����ʺϼ���������ݡ�

�㷨ʵ��
**********************************************************************************/
#include <fstream>
#include <cstring>
#define Maxm 501

using namespace std;

ifstream fin("APSP.in");
ofstream fout("APSP.out");

int p, q, k, m;
int Vertex, Line[Maxm];
int Path[Maxm][Maxm], Map[Maxm][Maxm], Dist[Maxm][Maxm];

void Root(int p, int q)
{
    if (Path[p][q] > 0)
    {
        Root(p, Path[p][q]);
        Root(Path[p][q], q);
    }
    else
    {
        Line[k] = q;
        k++;
    }
}
//�޷���ͨ��������֮�����Ϊ0
int main()
{
    memset(Path, 0, sizeof(Path));
    memset(Map, 0, sizeof(Map));
    memset(Dist, 0, sizeof(Dist));

    fin >> Vertex;
    for (p = 1; p <= Vertex; p++)
        for (q = 1; q <= Vertex; q++)
        {
            fin >> Map[p][q];
            Dist[p][q] = Map[p][q];
        }
    for (k = 1; k <= Vertex; k++)
        for (p = 1; p <= Vertex; p++)
            if (Dist[p][k] > 0)
                for (q = 1; q <= Vertex; q++)
                    if (Dist[k][q] > 0)
                    {
                        if (((Dist[p][q] > Dist[p][k] + Dist[k][q]) || (Dist[p][q] == 0)) && (p != q))
                        {
                            Dist[p][q] = Dist[p][k] + Dist[k][q];
                            Path[p][q] = k;
                        }
                    }

    for (p = 1; p <= Vertex; p++)
    {
        for (q = p + 1; q <= Vertex; q++)
        {
            fout << "\n==========================\n";
            fout << "Source:" << p << '\n'
                 << "Target " << q << '\n';
            fout << "Distance:" << Dist[p][q] << '\n';
            fout << "Path:" << p;
            k = 2;
            Root(p, q);
            for (m = 2; m <= k - 1; m++)
                fout << "-->" << Line[m];
            fout << '\n';
            fout << "==========================\n";
        }
    }
    fin.close();
    fout.close();
    return 0;
}
