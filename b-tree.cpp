#include<bits/stdc++.h>
using namespace std;
#define Td 3
typedef struct node{
    struct node** children;
    int* keys;
    int keynum;
    bool Leaf;
}node;

node* allocate_node(){
    node* newnode = (node*)malloc(sizeof(node));
    newnode->Leaf = true;
    newnode->keynum = 0;
    newnode->keys = (int*)malloc(sizeof(int)*(2*Td - 1));
    newnode->children = (node**)malloc(sizeof(node*) * Td * 2);
    return newnode;
}

void Bsplitchild(node* x,int i){
    node* y = x->children[i];
    node* half = allocate_node();
    half->keynum = Td - 1;
    for(int k = 0;k<Td - 1;k++){
        half->keys[k] = y->keys[k+Td];
    }
    half->Leaf = y->Leaf;
    if(!y->Leaf){
        for(int k = 0;k<Td;k++){
            half->children[k] = y->children[Td + k];
        }
    }
    y->keynum = Td;
    for(int k = x->keynum-1;k>=i;k--){
        x->keys[k+1] = x->keys[k];
    }
    for(int k = x->keynum;k>i;k--){
        x->children[k+1] = x->children[k];
    }
    x->keys[i] = y->keys[Td-1];
    x->keynum = x->keynum + 1;
    y->keynum = Td -1;
    x->children[i+1] = half;
}

void Bsplitroot(node** head){
    node* s = allocate_node();
    s->Leaf = false;
    s->children[0] = *head;
    Bsplitchild(s,0);
    *head = s;
}

void InsertnonFull(node* x,int key){
    if(x->Leaf){
        int i = x->keynum-1;
        while(x->keys[i] > key && i>-1){
            x->keys[i+1] = x->keys[i];
            i--;
        }
        i++;
        x->keys[i] = key;
        x->keynum = x->keynum + 1;
    }
    else{
        int i = x->keynum-1;
        while(x->keys[i] > key && i>-1){
            i--;
        }
        i++;
        if((x->children[i])->keynum == 2*Td - 1){
            Bsplitchild(x,i);
            if(key>x->keys[i]){
                i++;
            }
        }
        InsertnonFull(x->children[i],key);
    }
}

void PrintNode(node* n){
    cout<<"node\n";
    for(int i = 0;i<n->keynum;i++){
        cout<<n->keys[i]<<" ";
    }
    cout<<endl;

    for(int i = 0;i<=n->keynum && n->Leaf == false;i++){
        PrintNode(n->children[i]);
    }
}

void Btree_insert(node** head,int key){
    node* r = *head;
    if(r->keynum == 2*Td - 1){
        Bsplitroot(head);
    }
    InsertnonFull(*head,key);
}
int main(){
    node** head = (node**)malloc(sizeof(node*));
    *head = allocate_node();

    PrintNode(*head);

    while (true)
    {
        int key;
        cin>>key;
        Btree_insert(head,key);
        PrintNode(*head);
        cout<<"tree end \n";
    }
}